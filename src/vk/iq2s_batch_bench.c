// iq2s_batch_bench.c — 批量多矩阵 kernel 对拍：2 个 [1024,2560] IQ2_S 矩阵共享 h
// 期望 y[0..1023] = W_A·h, y[1024..2047] = W_B·h（W_B = kern13 参照逐元素核对）
// gcc -O2 iq2s_batch_bench.c -o iq2s_batch_bench -lvulkan -ldl -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <math.h>
#include "iq2s_grid_llama.h"

#define CHECK(expr, msg) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "VkError %d @ %s\n", _r, msg); exit(1); } } while (0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const int NB = 10, NIN = 2560, NOUT = 1024, NMAT = 2;
    const long ROWBYTES = (long)NB * 82;
    const long MAT_BYTES = (long)NOUT * ROWBYTES;

    // 权重 blob：vexp0 行 0-1023（矩阵 A）+ 行 1024-2047（矩阵 B）
    FILE* f = fopen("vexp0.iq2s.bin", "rb");
    if (!f) { perror("vexp0"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    long wblob = 2 * MAT_BYTES;
    if (fsz < wblob) { fprintf(stderr, "vexp0 太小\n"); return 1; }
    rewind(f);
    uint8_t* W = malloc(wblob);
    if (fread(W, 1, wblob, f) != (size_t)wblob) return 1;
    fclose(f);

    float* h = malloc(NIN * 4);
    srand(11);
    for (int i = 0; i < NIN; i++) h[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    // ── Vulkan init（紧凑）──
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst), "inst");
    VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &(uint32_t){1}, &pd);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
    int qf = -1;
    for (uint32_t i = 0; i < qn; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = (int)i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = (uint32_t)qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev), "dev");
    VkQueue queue; vkGetDeviceQueue(dev, (uint32_t)qf, 0, &queue);

    int type_h0 = -1;   // heap0 HV|COHERENT
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (!(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
        if (mp.memoryHeaps[mp.memoryTypes[i].heapIndex].flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) continue;
        type_h0 = (int)i; break;
    }
    printf("heap0 类型 = %d\n", type_h0);

    // 缓冲：0=W blob (heap0 直写) 1=GRID16 2=X 3=Y 4=MT
    VkBuffer bufs[5]; VkDeviceMemory mems[5]; void* maps[5];
    VkDeviceSize sizes[5] = { wblob, 1024 * 4, NIN * 4, 2 * NOUT * 4, 64 * 4 };
    const void* srcs[5] = { W, NULL, h, NULL, NULL };
    for (int i = 0; i < 5; i++) {
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = sizes[i], .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                             (i == 0 ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0) };
        CHECK(vkCreateBuffer(dev, &bci, NULL, &bufs[i]), "buf");
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        int mt = -1;
        for (uint32_t k = 0; k < mp.memoryTypeCount; k++) {
            if (!(mr.memoryTypeBits & (1u << k))) continue;
            VkMemoryPropertyFlags fl = mp.memoryTypes[k].propertyFlags;
            if (!(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || !(fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
            if (mp.memoryHeaps[mp.memoryTypes[k].heapIndex].size < sizes[i]) continue;
            mt = (int)k; break;
        }
        if (mt < 0) { fprintf(stderr, "无内存类型 buf%d\n", i); return 1; }
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = sizes[i], .memoryTypeIndex = (uint32_t)mt };
        CHECK(vkAllocateMemory(dev, &mai, NULL, &mems[i]), "alloc");
        CHECK(vkBindBufferMemory(dev, bufs[i], mems[i], 0), "bind");
        CHECK(vkMapMemory(dev, mems[i], 0, sizes[i], 0, &maps[i]), "map");
        if (srcs[i]) memcpy(maps[i], srcs[i], (size_t)sizes[i]);
    }
    // MT 表（uvec4 × 矩阵）：{w_off_words, y_off_rows, n_out, x_off_floats}
    uint32_t* mt = (uint32_t*)maps[4];
    memset(mt, 0, 64 * 4);
    // 矩阵 0: {w_off=0, y_off=0, n_out=1024, x_off=0}
    mt[0] = 0;        mt[1] = 0;      mt[2] = NOUT;   mt[3] = 0;
    // 矩阵 1: {w_off=MAT_BYTES/4, y_off=1024, n_out=1024, x_off=0}
    mt[4] = MAT_BYTES / 4; mt[5] = NOUT; mt[6] = NOUT; mt[7] = 0;
    memset(maps[3], 0, sizes[3]);

    // spv + pipeline
    FILE* sp = fopen("iq2s_batch.spv", "rb");
    if (!sp) { perror("spv"); return 1; }
    fseek(sp, 0, SEEK_END); long spsz = ftell(sp); rewind(sp);
    char* code = malloc((size_t)spsz);
    if (fread(code, 1, (size_t)spsz, sp) != (size_t)spsz) return 1;
    fclose(sp);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = (size_t)spsz, .pCode = (uint32_t*)code };
    VkShaderModule smod; CHECK(vkCreateShaderModule(dev, &smci, NULL, &smod), "sm");

    VkDescriptorPoolSize dpsz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &dpsz };
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool), "dpool");
    VkDescriptorSetLayoutBinding lb[5] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = 5, .pBindings = lb };
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl), "dsl");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet dset; CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset), "dset");
    VkDescriptorBufferInfo dbi[5];
    for (int i = 0; i < 5; i++) dbi[i] = (VkDescriptorBufferInfo){ bufs[i], 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet wr[5];
    for (int i = 0; i < 5; i++)
        wr[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = dset, .dstBinding = (uint32_t)i,
                                        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        .pBufferInfo = &dbi[i] };
    vkUpdateDescriptorSets(dev, 5, wr, 0, NULL);
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &dsl,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout playout; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &playout), "pl");
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                           .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = smod, .pName = "main" };
    VkComputePipelineCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                        .stage = ss, .layout = playout };
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, NULL, 1, &pci, NULL, &pipe), "pipe");
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = (uint32_t)qf };
    VkCommandPool cpool; CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cpool), "cpool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd), "cmd");
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CHECK(vkCreateFence(dev, &fci, NULL, &fence), "fence");

    // push: {n_mat, nb, 0, 0}
    uint32_t pc[4] = { 2, NB, 0, 0 };
    double tbest = 1e9;
    for (int rep = 0; rep < 8; rep++) {
        CHECK(vkResetCommandBuffer(cmd, 0), "rc");
        VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CHECK(vkBeginCommandBuffer(cmd, &bbi), "begin");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, NULL);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
        vkCmdDispatch(cmd, 2 * NOUT, 1, 1);
        CHECK(vkEndCommandBuffer(cmd), "end");
        CHECK(vkResetFences(dev, 1, &fence), "rf");
        double t0 = now_s();
        CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                                       .commandBufferCount = 1, .pCommandBuffers = &cmd }, fence), "submit");
        CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait");
        double dt = now_s() - t0;
        if (dt < tbest) tbest = dt;
    }
    float* ygpu = (float*)maps[3];

    // CPU 参照：kern13 逐矩阵
    void* lib = dlopen("/media/xiao_/OverSys1/npu-direct/hybrid/m5/m5_kern13.so", RTLD_NOW);
    int (*k13)(int, const float*, const uint8_t*, int, int, float*) =
        (int (*)(int, const float*, const uint8_t*, int, int, float*))dlsym(lib, "m5_gemv");
    float* yc = malloc(2 * NOUT * 4);
    k13(13, h, W, NOUT, NIN, yc);
    k13(13, h, W + MAT_BYTES / 4, NOUT, NIN, yc + NOUT);
    printf("GPU y[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", ygpu[i]);
    printf("\nCPU y[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", yc[i]);
    printf("\n");
    float maxd = 0;
    for (int i = 0; i < 2 * NOUT; i++) {
        float d = fabsf(ygpu[i] - yc[i]);
        if (d > maxd) maxd = d;
    }
    printf("批量 kernel vs kern13: max|Δ|=%.3e  %s\n", maxd, maxd < 1e-3 ? "PASS ✓" : "FAIL");
    printf("GPU 时间(2×NOUT 行): %.3f ms\n", tbest * 1000);
    return 0;
}
