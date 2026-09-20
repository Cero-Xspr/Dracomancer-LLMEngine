// vulkan_probe2.c — F1a v2：堆拆分容量测试 + iGPU 全量读带宽
// heap1 (DEV_LOCAL|HOST_VISIBLE|COHERENT, 8.44GB 预算) 装前 8.4G；
// heap0 (HOST_VISIBLE|COHERENT, 4.22GB 预算) 装余量。kernel 双缓冲索引。
// gcc -O2 vulkan_probe2.c -o vk_probe2 -lvulkan   （先 glslc probe_kern2.comp -o probe_kern2.spv）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(expr, msg) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "VkError %d @ %s\n", _r, msg); exit(1); } } while (0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void* map; long size; int type;
} BigBuf;

// 在指定内存类型序列里分配 size 字节并绑定；返回 VkResult
static VkResult alloc_buf(VkDevice dev, VkPhysicalDeviceMemoryProperties* mp,
                          VkBufferUsageFlags usage, long size,
                          const int* pref_types, int n_pref,
                          BigBuf* out) {
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = (VkDeviceSize)size, .usage = usage };
    CHECK(vkCreateBuffer(dev, &bci, NULL, &out->buf), "create buf");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, out->buf, &mr);
    int mtype = -1;
    for (int k = 0; k < n_pref && mtype < 0; k++) {
        for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
            if (!(mr.memoryTypeBits & (1u << i))) continue;
            if (mp->memoryTypes[i].propertyFlags != (VkMemoryPropertyFlags)pref_types[k]) continue;
            if (mp->memoryHeaps[mp->memoryTypes[i].heapIndex].size < (VkDeviceSize)size) continue;
            mtype = (int)i; break;
        }
    }
    if (mtype < 0) {
        for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
            if (!(mr.memoryTypeBits & (1u << i))) continue;
            if (!(mp->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            if (mp->memoryHeaps[mp->memoryTypes[i].heapIndex].size < (VkDeviceSize)size) continue;
            mtype = (int)i; break;
        }
    }
    if (mtype < 0) { vkDestroyBuffer(dev, out->buf, NULL); return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
    out->type = mtype;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = mr.size, .memoryTypeIndex = (uint32_t)mtype };
    VkResult r = vkAllocateMemory(dev, &mai, NULL, &out->mem);
    if (r != VK_SUCCESS) { vkDestroyBuffer(dev, out->buf, NULL); return r; }
    CHECK(vkBindBufferMemory(dev, out->buf, out->mem, 0), "bind");
    CHECK(vkMapMemory(dev, out->mem, 0, (VkDeviceSize)size, 0, &out->map), "map");
    out->size = size;
    return VK_SUCCESS;
}

int main(int argc, char** argv) {
    const char* gguf_path = (argc > 1) ? argv[1]
        : "/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-IQ2_M.gguf";

    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &ai };
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst), "instance");
    uint32_t nd = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
    int qf = -1;
    for (uint32_t i = 0; i < qn; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = (int)i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = (uint32_t)qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev), "device");
    VkQueue queue; vkGetDeviceQueue(dev, (uint32_t)qf, 0, &queue);

    FILE* f = fopen(gguf_path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", gguf_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    printf("权重: %.2f GB\n", fsz / 1e9);

    // ── 堆拆分：先 heap1(DEV_LOCAL|HV|C=0x7) 装满预算，余量 heap0(HV|C=0x6) ──
    long szA = fsz > 8400000000L ? 8400000000L : fsz;   // ★ F1a-v3：只用 heap1（VRAM 域）
    long szB = 0;                                        //   避开 GTT 域提交墙（无根解法）
    BigBuf A = {0}, B = {0};
    int pref_h1[1] = { 0x7 };           // DEV_LOCAL|HOST_VISIBLE|HOST_COHERENT
    int pref_h0[1] = { 0x6 };           // HOST_VISIBLE|HOST_COHERENT
    VkResult ra = alloc_buf(dev, &mp, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, szA,
                            (szA > 0) ? pref_h1 : pref_h0, 1, &A);
    if (ra != VK_SUCCESS) { fprintf(stderr, "heap1 分配 %.2fGB 失败 (%d)\n", szA / 1e9, ra); return 2; }
    printf("bufA %.2f GB @type%d(heap%u) OK\n", szA / 1e9, A.type, mp.memoryTypes[A.type].heapIndex);
    if (szB > 0) {
        VkResult rb = alloc_buf(dev, &mp, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, szB, pref_h0, 1, &B);
        if (rb != VK_SUCCESS) { fprintf(stderr, "heap0 分配 %.2fGB 失败 (%d)\n", szB / 1e9, rb); return 2; }
        printf("bufB %.2f GB @type%d(heap%u) OK\n", szB / 1e9, B.type, mp.memoryTypes[B.type].heapIndex);
    }
    printf("⇒ 容量测试通过：%.2f GB 全部 GPU 可见\n", fsz / 1e9);

    // CPU 侧灌入真权重
    double t0 = now_s();
    rewind(f);
    long done = fread(A.map, 1, szA, f);
    if (szB > 0 && done == szA) done += fread(B.map, 1, szB, f);
    fclose(f);
    printf("CPU 读权重进缓冲: %.2f GB in %.2fs (%.1f GB/s)\n", done / 1e9, now_s() - t0, done / (now_s() - t0) / 1e9);

    // ── shader ──
    FILE* sp = fopen("probe_kern2.spv", "rb");
    if (!sp) { fprintf(stderr, "缺 probe_kern2.spv\n"); return 1; }
    fseek(sp, 0, SEEK_END); long spsz = ftell(sp); rewind(sp);
    char* code = malloc((size_t)spsz); if (fread(code, 1, (size_t)spsz, sp) != (size_t)spsz) return 1; fclose(sp);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = (size_t)spsz, .pCode = (uint32_t*)code };
    VkShaderModule smod; CHECK(vkCreateShaderModule(dev, &smci, NULL, &smod), "shader");

    const uint32_t NINV = 65536;
    VkBuffer obuf; VkDeviceMemory omem;
    VkBufferCreateInfo obci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = NINV * 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    CHECK(vkCreateBuffer(dev, &obci, NULL, &obuf), "obuf");
    VkMemoryRequirements omr; vkGetBufferMemoryRequirements(dev, obuf, &omr);
    int otype = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((omr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { otype = (int)i; break; }
    VkMemoryAllocateInfo omai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize = omr.size, .memoryTypeIndex = (uint32_t)otype };
    CHECK(vkAllocateMemory(dev, &omai, NULL, &omem), "omem");
    CHECK(vkBindBufferMemory(dev, obuf, omem, 0), "obind");

    VkDescriptorPoolSize dpsz[1] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 } };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = dpsz };
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool), "dpool");
    VkDescriptorSetLayoutBinding lb[3] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = 3, .pBindings = lb };
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl), "dsl");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet dset; CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset), "dset");
    VkDescriptorBufferInfo dbi[3] = {
        { A.buf, 0, VK_WHOLE_SIZE }, { szB ? B.buf : A.buf, 0, VK_WHOLE_SIZE }, { obuf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wr[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = 2,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[2] } };
    vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);

    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 8 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &dsl,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout playout; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &playout), "playout");
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                           .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = smod, .pName = "main" };
    VkComputePipelineCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                        .stage = ss, .layout = playout };
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, NULL, 1, &pci, NULL, &pipe), "pipeline");

    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .queueFamilyIndex = (uint32_t)qf };
    VkCommandPool cpool; CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cpool), "cpool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd), "cmd");
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CHECK(vkCreateFence(dev, &fci, NULL, &fence), "fence");

    uint32_t countA = (uint32_t)(szA / 4);
    uint32_t count = (uint32_t)(done / 4);
    printf("\niGPU 全量读 %.2f GB × 3 遍（countA=%u count=%u）...\n", done / 1e9, countA, count);
    struct { uint32_t a, b; } pcvals = { countA, count };

    double tg0 = now_s();
    for (int rep = 0; rep < 3; rep++) {
        fprintf(stderr, "[rep%d] begin\n", rep);
        CHECK(vkResetCommandBuffer(cmd, 0), "reset");
        VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CHECK(vkBeginCommandBuffer(cmd, &bbi), "begin");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, NULL);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcvals), &pcvals);
        vkCmdDispatch(cmd, NINV / 256, 1, 1);
        CHECK(vkEndCommandBuffer(cmd), "end");
        CHECK(vkResetFences(dev, 1, &fence), "fence reset");
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1, .pCommandBuffers = &cmd };
        CHECK(vkQueueSubmit(queue, 1, &si, fence), "submit");
        double tw = now_s();
        CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait");
        fprintf(stderr, "[rep%d] wait %.1f ms\n", rep, (now_s()-tw)*1000);
    }
    double dt = (now_s() - tg0) / 3.0;
    long bytes = done;
    printf("GPU 读带宽: %.2f GB in %.3fs = %.1f GB/s\n", bytes / 1e9, dt, bytes / dt / 1e9);

    uint32_t* om = NULL; CHECK(vkMapMemory(dev, omem, 0, NINV * 4, 0, (void**)&om), "omap");
    uint32_t nz = 0; for (uint32_t i = 0; i < NINV; i++) if (om[i]) nz++;
    printf("校验（非零 invocation）: %u/%u\n", nz, NINV);

    vkDestroyFence(dev, fence, NULL); vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyPipeline(dev, pipe, NULL); vkDestroyPipelineLayout(dev, playout, NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL); vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyShaderModule(dev, smod, NULL);
    vkFreeMemory(dev, omem, NULL); vkDestroyBuffer(dev, obuf, NULL);
    vkFreeMemory(dev, A.mem, NULL); vkDestroyBuffer(dev, A.buf, NULL);
    if (szB > 0) { vkFreeMemory(dev, B.mem, NULL); vkDestroyBuffer(dev, B.buf, NULL); }
    vkDestroyDevice(dev, NULL); vkDestroyInstance(inst, NULL);
    printf("F1a v2 探针完成\n");
    return 0;
}
