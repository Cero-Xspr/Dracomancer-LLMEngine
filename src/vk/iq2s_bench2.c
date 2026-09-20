// iq2s_bench.c — F1b：GPU IQ2_S gemv vs CPU kern13 对拍 + 计速
// gcc -O2 iq2s_bench.c -o iq2s_bench -lvulkan
//   （先 glslc iq2s_gemv.comp -o iq2s_gemv.spv；GRIDSGN_F 来自 iq2s_grid_f32.h）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <math.h>
// GRID16 用 g16.bin（4KB uint 码表）

#define CHECK(expr, msg) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "VkError %d @ %s\n", _r, msg); exit(1); } } while (0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// CPU 参照：kern13 的 m5_gemv
static int (*m5_ref)(int, const float*, const uint8_t*, int, int, float*);
static void load_ref(void) {
    void* lib = dlopen("/media/xiao_/OverSys1/npu-direct/hybrid/m5/m5_kern13.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); exit(1); }
    m5_ref = (int (*)(int, const float*, const uint8_t*, int, int, float*))dlsym(lib, "m5_gemv");
    if (!m5_ref) { fprintf(stderr, "无 m5_gemv\n"); exit(1); }
}

int main(void) {
    const int NB = 10;            // 2560/256
    const int NIN = NB * 256;     // 2560
    const int NOUT = 1024;        // 一个 v_expert 的行数
    const long ROWBYTES = (long)NB * 82;

    // 权重：从 IQ2_M 的 blk.3.attn_v_exps 取专家 0
    FILE* f = fopen("/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-IQ2_M.gguf", "rb");
    if (!f) { fprintf(stderr, "no model\n"); return 1; }
    // 专家 0 在该张量的偏移 = 12,459,958,272 - ... 太绕：直接用 gguf_fast 读
    // 这里省事：走 python 导出的切片文件（若无则提示生成）
    const char* slice = "vexp0.iq2s.bin";
    f = fopen(slice, "rb");
    if (!f) { fprintf(stderr, "缺 %s（用 python 导出 v_expert0 原始字节）\n", slice); return 1; }
    fseek(f, 0, SEEK_END);
    long wbytes = ftell(f); rewind(f);
    uint8_t* W = malloc((size_t)wbytes);
    if (fread(W, 1, (size_t)wbytes, f) != (size_t)wbytes) return 1;
    fclose(f);
    printf("权重 %ld B（%d 行 × %ld B）\n", wbytes, NOUT, ROWBYTES);

    float* h = malloc(NIN * 4);
    srand(7);
    int UNIT = getenv("UNIT") ? atoi(getenv("UNIT")) : -1;
    if (UNIT >= 0) {
        memset(h, 0, NIN * 4); h[UNIT] = 1.0f;
    } else {
        for (int i = 0; i < NIN; i++) h[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }

    // ── Vulkan 初始化（紧凑版）──
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

    int type_hv_devlocal = -1, type_hv = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
        if (!(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if ((fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            && type_hv_devlocal < 0) type_hv_devlocal = (int)i;
        if ((fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && type_hv < 0) type_hv = (int)i;
    }

    // 缓冲：W / GRID / X / Y
    VkBuffer bufs[4]; VkDeviceMemory mems[4];
    VkDeviceSize sizes[4] = { wbytes, 1024 * 4, NIN * 4, NOUT * 4 };
    static uint32_t g16raw[1024];   // ★ GPU 侧按 uint 索引，必须扩成 uint32
    {
        FILE* gf = fopen("g16.bin", "rb");
        if (!gf) { fprintf(stderr, "缺 g16.bin\n"); return 1; }
        uint16_t tmp[1024];
        if (fread(tmp, 2, 1024, gf) != 1024) return 1;
        fclose(gf);
        for (int i = 0; i < 1024; i++) g16raw[i] = tmp[i];
    }
    int types[4] = { type_hv_devlocal, type_hv_devlocal, type_hv, type_hv };
    void* maps[4];
    const void* srcs[4] = { W, g16raw, h, NULL };
    for (int i = 0; i < 4; i++) {
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = sizes[i], .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
        CHECK(vkCreateBuffer(dev, &bci, NULL, &bufs[i]), "buf");
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        int mt = -1;
        for (uint32_t k = 0; k < mp.memoryTypeCount; k++) {
            if (!(mr.memoryTypeBits & (1u << k))) continue;
            VkMemoryPropertyFlags fl = mp.memoryTypes[k].propertyFlags;
            if (!(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || !(fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
        if (i < 2 && !(fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;   // W/GRID 必须 DEV_LOCAL（heap0 写入对 GPU 不可见的坑）    
            if (mp.memoryHeaps[mp.memoryTypes[k].heapIndex].size < sizes[i]) continue;
            mt = (int)k; break;
        }
        if (mt < 0) mt = types[i];
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = sizes[i], .memoryTypeIndex = (uint32_t)mt };
        CHECK(vkAllocateMemory(dev, &mai, NULL, &mems[i]), "mem");
        CHECK(vkBindBufferMemory(dev, bufs[i], mems[i], 0), "bind");
        CHECK(vkMapMemory(dev, mems[i], 0, sizes[i], 0, &maps[i]), "map");
        if (srcs[i]) {
            memcpy(maps[i], srcs[i], (size_t)sizes[i]);
            int cpu_rb = memcmp(maps[i], srcs[i], (size_t)(sizes[i] > 64 ? 64 : sizes[i])) == 0;
            fprintf(stderr, "[upload %d] CPU 回读一致=%d  type=%d\n", i, cpu_rb, types[i]);
        }
    }

    FILE* sp = fopen(getenv("DIAG") ? "iq2s_diag.spv" : "iq2s_gemv.spv", "rb");
    if (!sp) { fprintf(stderr, "缺 iq2s_gemv.spv\n"); return 1; }
    fseek(sp, 0, SEEK_END); long spsz = ftell(sp); rewind(sp);
    char* code = malloc((size_t)spsz); if (fread(code, 1, (size_t)spsz, sp) != (size_t)spsz) return 1; fclose(sp);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = (size_t)spsz, .pCode = (uint32_t*)code };
    VkShaderModule smod; CHECK(vkCreateShaderModule(dev, &smci, NULL, &smod), "sm");
    VkDescriptorPoolSize dpsz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &dpsz };
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool), "dpool");
    VkDescriptorSetLayoutBinding lb[4] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = 4, .pBindings = lb };
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl), "dsl");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet dset; CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset), "dset");
    VkDescriptorBufferInfo dbi[4];
    for (int i = 0; i < 4; i++) dbi[i] = (VkDescriptorBufferInfo){ bufs[i], 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet wr[4];
    for (int i = 0; i < 4; i++)
        wr[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = dset, .dstBinding = (uint32_t)i,
                                        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        .pBufferInfo = &dbi[i] };
    vkUpdateDescriptorSets(dev, 4, wr, 0, NULL);
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &dsl,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout playout; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &playout), "pl");
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                           .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = smod, .pName = "main" };
    VkComputePipelineCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                        .stage = ss, .layout = playout };
    VkResult pipe_res[1] = { VK_SUCCESS };
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, NULL, 1, &pci, NULL, &pipe), "pipe");
    if (pipe == VK_NULL_HANDLE || pipe_res[0] != VK_SUCCESS) {
        fprintf(stderr, "*** pipeline 创建失败: pipe=%p res=%d\n", (void*)pipe, pipe_res[0]);
        return 3;
    }
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = (uint32_t)qf };
    VkCommandPool cpool; CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cpool), "cpool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd), "cmd");
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CHECK(vkCreateFence(dev, &fci, NULL, &fence), "fence");

    uint32_t pcv[2] = { (uint32_t)NOUT, (uint32_t)NB };
    if (getenv("DIAG")) { pcv[0] = 12; pcv[1] = 10; }
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    double tbest = 1e9;
    for (int rep = 0; rep < 8; rep++) {
        CHECK(vkResetCommandBuffer(cmd, 0), "rc");
        VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CHECK(vkBeginCommandBuffer(cmd, &bbi), "begin");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, NULL);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pcv);
        vkCmdDispatch(cmd, getenv("DIAG") ? 12u : (uint32_t)NOUT, 1, 1);
        CHECK(vkEndCommandBuffer(cmd), "end");
        CHECK(vkResetFences(dev, 1, &fence), "rf");
        double t0 = now_s();
        CHECK(vkQueueSubmit(queue, 1, &si, fence), "submit");
        CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait");
        double dt = now_s() - t0;
        if (dt < tbest) tbest = dt;
    }
    float* ymap = (float*)maps[3];
    // CPU 参照
    load_ref();
    float* ycpu = malloc(NOUT * 4);
    m5_ref(13, h, W, NOUT, NIN, ycpu);
    double t0 = now_s();
    for (int rep = 0; rep < 100; rep++) m5_ref(13, h, W, NOUT, NIN, ycpu);
    double tcpu = (now_s() - t0) / 100;
    float maxd = 0; int nbad = 0;
    for (int i = 0; i < NOUT; i++) {
        float d = fabsf(ymap[i] - ycpu[i]);
        if (d > 1e-3f) nbad++;
        if (d > maxd) maxd = d;
    }
    if (UNIT >= 0) {
        printf("UNIT=%d: GPU y[0..5] =", UNIT);
        for (int i = 0; i < 6; i++) printf(" %.4f", ymap[i]);
        printf("\n         CPU y[0..5] =");
        for (int i = 0; i < 6; i++) printf(" %.4f", ycpu[i]);
        printf("\n");
    }
    if (getenv("DIAG")) {
        for (int i = 0; i < 768; i++)
            printf("DIAG %d = %.6g\n", i, ymap[i]);
        return 0;
    }
    printf("GPU y[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", ymap[i]);
    printf("\nCPU y[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", ycpu[i]);
    printf("\n");
    printf("\niGPU gemv: %.3f ms (%.1f GB/s)   CPU kern13: %.3f ms (%.1f GB/s)\n",
           tbest * 1000, wbytes / 1e9 / tbest, tcpu * 1000, wbytes / 1e9 / tcpu);
    printf("对拍: max|Δ|=%.3e  超差行=%d/%d  %s\n", maxd, nbad, NOUT,
           (maxd < 1e-3 && nbad == 0) ? "PASS" : "FAIL");
    return 0;
}
