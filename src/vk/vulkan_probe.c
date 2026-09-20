// vulkan_probe.c — F1a：RADV UMA 大缓冲容量 + iGPU 全量读带宽探针
// 1) 报告堆/最大分配限制  2) 把 IQ2_M 真权重读进 host-visible 缓冲（12.4GB 容量测试）
// 3) 派发 sum-reduce 让 iGPU 读完全部数据，测 GPU 侧读带宽
// gcc -O2 vulkan_probe.c -o vk_probe -lvulkan
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

int main(int argc, char** argv) {
    const char* gguf_path = (argc > 1) ? argv[1]
        : "/media/Data-1/gguf/k2-horizon/K2-Horizon-MoVA-36B-A4B-IQ2_M.gguf";

    // ── 实例与物理设备 ──
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &ai };
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst), "instance");
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice devs[8]; vkEnumeratePhysicalDevices(inst, &nd, devs);
    VkPhysicalDevice pd = devs[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("设备: %s (vendor 0x%x, api %u.%u)\n", props.deviceName,
           props.vendorID, VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion));
    printf("maxMemoryAllocationCount = %u, maxStorageBufferRange = %u\n",
           props.limits.maxMemoryAllocationCount, props.limits.maxStorageBufferRange);

    // ── 内存堆 ──
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("内存堆 %u 个:\n", mp.memoryHeapCount);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        printf("  heap[%u] size=%.2f GB flags=%s%s\n", i,
               mp.memoryHeaps[i].size / 1e9,
               (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
               (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) ? "SPERSIST" : "");
    printf("内存类型 %u 个:\n", mp.memoryTypeCount);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        printf("  type[%u] heap=%u flags=0x%x%s%s%s\n", i, mp.memoryTypes[i].heapIndex,
               mp.memoryTypes[i].propertyFlags,
               (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
               (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
               (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEV_LOCAL" : "");

    // ── 队列族（找 compute）──
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
    int qf = -1;
    for (uint32_t i = 0; i < qn; i++)
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = (int)i; break; }
    if (qf < 0) { fprintf(stderr, "无 compute 队列\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = (uint32_t)qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev), "device");
    VkQueue queue; vkGetDeviceQueue(dev, (uint32_t)qf, 0, &queue);

    // ── 大缓冲：host-visible|coherent，装下整个 GGUF ──
    FILE* f = fopen(gguf_path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", gguf_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    printf("\n权重文件: %s (%.2f GB)\n", gguf_path, fsz / 1e9);

    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = (VkDeviceSize)fsz,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VkBuffer wbuf; CHECK(vkCreateBuffer(dev, &bci, NULL, &wbuf), "buffer");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, wbuf, &mr);
    printf("buffer 要求: size=%.2f GB align=%zu\n", mr.size / 1e9, (size_t)mr.alignment);
    int mtype = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mtype = (int)i; break; }
    if (mtype < 0) { fprintf(stderr, "无 host-visible 内存类型可容纳\n"); return 1; }
    printf("用内存类型 %d (heap %u)\n", mtype, mp.memoryTypes[mtype].heapIndex);

    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = mr.size, .memoryTypeIndex = (uint32_t)mtype };
    double ta = now_s();
    VkDeviceMemory wmem; VkResult ar = vkAllocateMemory(dev, &mai, NULL, &wmem);
    if (ar != VK_SUCCESS) { fprintf(stderr, "*** 12.4GB 单次分配失败 (%d)——容量上限场景确认\n", ar); return 2; }
    printf("分配 %.2f GB: OK (%.2fs)\n", mr.size / 1e9, now_s() - ta);
    CHECK(vkBindBufferMemory(dev, wbuf, wmem, 0), "bind");

    // CPU 侧写入真权重（读文件进映射内存 = F1 的装载原型）
    void* wmap; CHECK(vkMapMemory(dev, wmem, 0, mr.size, 0, &wmap), "map");
    double t0 = now_s();
    rewind(f);
    long done = 0;
    while (done < fsz) {
        size_t got = fread((char*)wmap + done, 1, 1 << 24, f);
        if (got == 0) break;
        done += (long)got;
    }
    fclose(f);
    printf("CPU 读文件进映射缓冲: %.2f GB in %.2fs (%.1f GB/s)\n",
           done / 1e9, now_s() - t0, done / (now_s() - t0) / 1e9);

    // ── 着色器模块 ──
    FILE* sp = fopen("probe_kern.spv", "rb");
    if (!sp) { fprintf(stderr, "缺 probe_kern.spv（先 glslc 编译）\n"); return 1; }
    fseek(sp, 0, SEEK_END); long spsz = ftell(sp); rewind(sp);
    char* code = malloc((size_t)spsz); fread(code, 1, (size_t)spsz, sp); fclose(sp);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = (size_t)spsz, .pCode = (uint32_t*)code };
    VkShaderModule smod; CHECK(vkCreateShaderModule(dev, &smci, NULL, &smod), "shader");

    // 输出缓冲：65536 调用 × uint32
    const uint32_t NINV = 65536;
    VkBuffer obuf; VkDeviceMemory omem;
    VkBufferCreateInfo obci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = NINV * 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    CHECK(vkCreateBuffer(dev, &obci, NULL, &obuf), "obuf");
    VkMemoryRequirements omr; vkGetBufferMemoryRequirements(dev, obuf, &omr);
    int omtype = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((omr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { omtype = (int)i; break; }
    VkMemoryAllocateInfo omai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize = omr.size, .memoryTypeIndex = (uint32_t)omtype };
    CHECK(vkAllocateMemory(dev, &omai, NULL, &omem), "omem");
    CHECK(vkBindBufferMemory(dev, obuf, omem, 0), "obind");

    // ── 描述符 ──
    VkDescriptorPoolSize dpsz[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 } };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 2, .poolSizeCount = 2, .pPoolSizes = dpsz };
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool), "dpool");
    VkDescriptorSetLayoutBinding lb[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = 2, .pBindings = lb };
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl), "dsl");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet dset; CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset), "dset");
    VkDescriptorBufferInfo dbi[2] = {
        { wbuf, 0, VK_WHOLE_SIZE }, { obuf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wr[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &dbi[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &dbi[1] } };
    vkUpdateDescriptorSets(dev, 2, wr, 0, NULL);

    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &dsl,
                                        .pushConstantRangeCount = 0 };
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

    // ── 记录：3 遍 dispatch（每遍全量读）──
    CHECK(vkResetCommandBuffer(cmd, 0), "reset");
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    CHECK(vkBeginCommandBuffer(cmd, &bbi), "begin");
    uint64_t count = (uint64_t)done / 4;   // uint32 元素数
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, NULL);
    vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint64_t), &count);
    vkCmdDispatch(cmd, NINV / 256, 1, 1);
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdDispatch(cmd, NINV / 256, 1, 1);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdDispatch(cmd, NINV / 256, 1, 1);
    CHECK(vkEndCommandBuffer(cmd), "end");

    printf("\nGPU 全量读 %ld 遍 × %.2f GB（%.2f 亿 uint32）...\n", 3, done / 1e9, count / 1e8);
    double tg0 = now_s();
    CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO }, fence), "submit");
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait");
    double dt = now_s() - tg0;
    long bytes = 3 * done;
    printf("GPU 读带宽: %.2f GB in %.2fs = %.1f GB/s\n", bytes / 1e9, dt, bytes / dt / 1e9);

    uint32_t* om = NULL; CHECK(vkMapMemory(dev, omem, 0, NINV * 4, 0, (void**)&om), "omap");
    uint32_t nonzero = 0; for (uint32_t i = 0; i < NINV; i++) if (om[i]) nonzero++;
    printf("输出校验（非零 invocation 数）: %u/%u\n", nonzero, NINV);

    vkDestroyFence(dev, fence, NULL); vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyPipeline(dev, pipe, NULL); vkDestroyPipelineLayout(dev, playout, NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL); vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyShaderModule(dev, smod, NULL); vkFreeMemory(dev, omem, NULL);
    vkDestroyBuffer(dev, obuf, NULL); vkFreeMemory(dev, wmem, NULL);
    vkDestroyBuffer(dev, wbuf, NULL); vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("\nF1a 探针完成\n");
    return 0;
}
