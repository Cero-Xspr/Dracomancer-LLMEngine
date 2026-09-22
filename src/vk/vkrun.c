// vkrun.c — F1c K2 iGPU 运行时：IQ2_S 权重 GPU 驻留 + 多矩阵单 submit gemv。
// 设计要点（F1b 实证）：
//   · heap1(DEV_LOCAL) 的 CPU mmap 写对 GPU 不可见 ⇒ 上传必须 staging+vkCmdCopyBuffer；
//     heap0(GTT|HOST_VISIBLE|COHERENT) 的 CPU 写 GPU 可读 ⇒ X/Y/G16 与溢出权重直接 memcpy。
//   · 权重 10.5GB > uint32 字节偏移 ⇒ 拆多个 ≤min(4GB,maxStorageBufferRange) 的 W 缓冲，
//     每缓冲一套 descriptor set（set i = {W[i], G16, X, Y}），dispatch 按 wset 选 set。
//   · 已知坑：VkSubmitInfo 必须 .pCommandBuffers=&cmd；pipeline layout 必须带 pushConstantRange。
// gcc -O2 -fopenmp -shared -fPIC vkrun.c -o libvkrun.so -lvulkan
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <vulkan/vulkan.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAXW 8
#define CHECK(expr, msg) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[vkrun] VkError %d @ %s\n", _r, msg); return -1; } } while (0)
// 锁内版本：失败先解锁（否则下次 submit 死锁）
#define CHECK2(expr, msg) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[vkrun] VkError %d @ %s\n", _r, msg); pthread_mutex_unlock(&vg->qlock); return -1; } } while (0)

struct VGMat { uint32_t w_off; uint32_t x_off; uint32_t y_off; uint32_t n_out; uint32_t nb; uint32_t wset; uint32_t pipe; };
struct VGInfo { int n_w; unsigned long w_bytes[MAXW]; int w_heap[MAXW]; void* w_map[MAXW]; };
struct VG {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue queue; uint32_t qf;
    VkDescriptorPool dpool; VkDescriptorSetLayout dsl; VkDescriptorSet dsets[MAXW];
    VkPipelineLayout playout; VkPipeline pipe; VkPipeline pipe2; int has_pipe2;
    VkPipeline pipe3; int has_pipe3;   // IQ2_S GEMM16（prefill 批处理）
    VkCommandPool cpool; VkCommandBuffer cmd; VkFence fence;
    int n_w; VkBuffer wb[MAXW]; VkDeviceMemory wm[MAXW]; unsigned long wbytes[MAXW]; int wheap[MAXW];
    void* wmap[MAXW];                       // 仅 GTT 缓冲非 NULL
    VkBuffer xb, yb, gb, sb; VkDeviceMemory xm, ym, gm, sm;
    void* xmap; void* ymap; void* gmap; void* smap;
    unsigned long xbytes, ybytes, sbytes;
    // keeper：DVFS 保活线程（CPU 相位期 GPU 空闲会掉频 ⇒ 持续小 dispatch 钉高频）
    pthread_mutex_t qlock;          // 队列互斥（keeper 与主线程的 submit 互斥）
    VkCommandBuffer kcmd; VkFence kfence;
    pthread_t keeper_thr; volatile int keeper_on; int keeper_started;
    VkQueue kqueue;                 // keeper 专用队列（NULL = 与主队列共享，走互斥+sleep）
    unsigned wg_rows;               // 每 WG 处理的行数（v5=4 ⇒ dispatch=ceil(n_out/4)）
};
typedef struct VG VG;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9; }

static int find_type(VG* vg, VkMemoryRequirements* mr, unsigned long size, int want_devlocal) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vg->pd, &mp);
    int best = -1;
    for (uint32_t k = 0; k < mp.memoryTypeCount; k++) {
        if (!(mr->memoryTypeBits & (1u << k))) continue;
        VkMemoryPropertyFlags fl = mp.memoryTypes[k].propertyFlags;
        if (mp.memoryHeaps[mp.memoryTypes[k].heapIndex].size < size) continue;
        int devlocal = (fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (want_devlocal) {
            // 权重缓冲：优先 DEV_LOCAL（不需要 host 可见——上传走 staging）
            if (devlocal && best < 0) best = (int)k;
        } else {
            // X/Y/G16/staging：必须 HOST_VISIBLE|COHERENT，且不选 DEV_LOCAL（CPU 写可见性）
            if ((fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                && !devlocal && best < 0) best = (int)k;
        }
    }
    return best;
}

static int mk_buf(VG* vg, VkBuffer* b, VkDeviceMemory* m, void** map,
                  unsigned long size, int want_devlocal, int type_hint) {
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT };
    CHECK(vkCreateBuffer(vg->dev, &bci, NULL, b), "mkbuf");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(vg->dev, *b, &mr);
    int mt = find_type(vg, &mr, size, want_devlocal);
    if (mt < 0) mt = type_hint;   // 回退：调用者保证合法
    if (mt < 0) { fprintf(stderr, "[vkrun] 无合适内存类型 size=%lu\n", size); return -1; }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = size, .memoryTypeIndex = (uint32_t)mt };
    CHECK(vkAllocateMemory(vg->dev, &mai, NULL, m), "alloc");
    CHECK(vkBindBufferMemory(vg->dev, *b, *m, 0), "bind");
    if (map) CHECK(vkMapMemory(vg->dev, *m, 0, size, 0, map), "map");
    return 0;
}

// keeper：持续提交 64 行哑 dispatch（读 W 起始的零区、写 Y 尾部 scratch），
// 把 amdgpu DVFS 钉在高频。与主线程共用 queue ⇒ 用 qlock 互斥。
static void* keeper_fn(void* p) {
    VG* vg = (VG*)p;
    uint32_t pc[5] = { 0, 0, (uint32_t)(vg->ybytes / 4) - 64u, 64u, 10u };
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &vg->kcmd };
    int shared_queue = (vg->kqueue == NULL);
    VkQueue q = shared_queue ? vg->queue : vg->kqueue;
    while (vg->keeper_on) {
        if (shared_queue) pthread_mutex_lock(&vg->qlock);
        if (!vg->keeper_on) { if (shared_queue) pthread_mutex_unlock(&vg->qlock); break; }
        vkResetCommandBuffer(vg->kcmd, 0);
        if (vkBeginCommandBuffer(vg->kcmd, &bbi) == VK_SUCCESS) {
            vkCmdBindPipeline(vg->kcmd, VK_PIPELINE_BIND_POINT_COMPUTE, vg->pipe);
            vkCmdBindDescriptorSets(vg->kcmd, VK_PIPELINE_BIND_POINT_COMPUTE, vg->playout,
                                    0, 1, &vg->dsets[0], 0, NULL);
            vkCmdPushConstants(vg->kcmd, vg->playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, pc);
            vkCmdDispatch(vg->kcmd, 64u, 1, 1);
            vkEndCommandBuffer(vg->kcmd);
            vkResetFences(vg->dev, 1, &vg->kfence);
            if (vkQueueSubmit(q, 1, &si, vg->kfence) == VK_SUCCESS)
                vkWaitForFences(vg->dev, 1, &vg->kfence, VK_TRUE, UINT64_MAX);
        }
        if (shared_queue) {
            pthread_mutex_unlock(&vg->qlock);
            usleep(500);            // ★ 共享队列模式必须留窗口，否则饿死主线程
        }
    }
    return NULL;
}

int vg_init(struct VG** out, unsigned long total_bytes, const char* spv_path,
            const char* spv2_path, const char* spv3_path, unsigned long x_bytes,
            unsigned long y_bytes, unsigned long stage_bytes, struct VGInfo* info) {
    *out = NULL;
    VG* vg = (VG*)calloc(1, sizeof(VG));
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    CHECK(vkCreateInstance(&ici, NULL, &vg->inst), "inst");
    vkEnumeratePhysicalDevices(vg->inst, &(uint32_t){1}, &vg->pd);
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(vg->pd, &pp);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(vg->pd, &mp);
    VkPhysicalDeviceLimits lim = pp.limits;
    fprintf(stderr, "[vkrun] GPU=%s maxStorageBufferRange=%.2fGB\n", pp.deviceName,
            lim.maxStorageBufferRange / 1073741824.0);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        fprintf(stderr, "[vkrun] heap%u %.2fGB\n", i, mp.memoryHeaps[i].size / 1073741824.0);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(vg->pd, &qn, NULL);
    VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(vg->pd, &qn, qp);
    int qf = -1;
    for (uint32_t i = 0; i < qn; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = (int)i; break; }
    vg->qf = (uint32_t)qf;
    float prio[2] = { 1.0f, 0.5f };
    uint32_t want_q = (qp[qf].queueCount >= 2) ? 2u : 1u;   // keeper 用第 2 条 queue，免锁
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = vg->qf, .queueCount = want_q,
                                    .pQueuePriorities = prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    CHECK(vkCreateDevice(vg->pd, &dci, NULL, &vg->dev), "dev");
    vkGetDeviceQueue(vg->dev, vg->qf, 0, &vg->queue);
    vg->kqueue = NULL;
    if (want_q >= 2u) vkGetDeviceQueue(vg->dev, vg->qf, 1, &vg->kqueue);

    // ── W 缓冲切分：≤min(4GB, maxStorageBufferRange)，heap1 优先 ──
    unsigned long cap = 4UL << 30;
    if (lim.maxStorageBufferRange > 0 && (unsigned long)lim.maxStorageBufferRange < cap)
        cap = (unsigned long)lim.maxStorageBufferRange;
    // ★ RADV：缓冲尺寸恰好等于 maxStorageBufferRange(4GB) 时 SSBO 描述子失效（读恒 0）
    //   —— 实测教训，上限留出余量
    if (cap > (3UL << 30)) cap = 3UL << 30;
    // DEV_LOCAL 堆容量（取最大 DEV_LOCAL 堆）
    unsigned long dl_heap = 0; int type_dl = -1, type_gtt = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
        unsigned long hs = mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size;
        if ((fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && hs > dl_heap) { dl_heap = hs; type_dl = (int)i; }
        if ((fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            && !(fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && type_gtt < 0) type_gtt = (int)i;
    }
    if (dl_heap > 0 && cap > dl_heap) cap = dl_heap;   // 单缓冲不超过 DEV_LOCAL 堆
    unsigned long dl_left = dl_heap ? (dl_heap - (8UL << 20)) : 0;   // 留 8MB 余量
    if (getenv("VKRUN_FORCE_GTT")) dl_left = 0;   // 实验：全部落 GTT（测堆带宽差）
    // GTT 预算：heap0 留出 X/Y/G16/staging（~0.19GB，stage=128MB）
    unsigned long gtt_budget = 0;
    if (type_gtt >= 0) {
        unsigned long hs = mp.memoryHeaps[mp.memoryTypes[type_gtt].heapIndex].size;
        gtt_budget = hs > (192UL << 20) ? hs - (192UL << 20) : 0;
    }
    const unsigned long MINCHUNK = 64UL << 20;
    vg->n_w = 0;
    unsigned long have = 0;
    while (have < total_bytes && vg->n_w < MAXW) {
        int dl = (dl_left >= MINCHUNK);
        unsigned long room = dl ? dl_left : gtt_budget;
        if (room < MINCHUNK) {
            fprintf(stderr, "[vkrun] 放不下：还差 %luB\n", total_bytes - have);
            return -1;
        }
        unsigned long sz = cap < room ? cap : room;   // 按容量分配，但留足装箱余量即可
        unsigned long need = total_bytes - have;
        if (sz > need + (128UL << 20)) sz = need + (128UL << 20);   // ★ 最后一档不超需求+128MB
                                                                    //   （省 GTT=RAM，防 OOM/swap-out）
        vg->wbytes[vg->n_w] = sz;
        vg->wheap[vg->n_w] = dl;
        int rc = mk_buf(vg, &vg->wb[vg->n_w], &vg->wm[vg->n_w],
                        dl ? NULL : &vg->wmap[vg->n_w], sz, dl ? 1 : 0, dl ? type_dl : type_gtt);
        if (rc != 0) return rc;
        if (dl) dl_left -= sz;
        else gtt_budget -= sz;
        have += sz;
        vg->n_w++;
    }
    if (have < total_bytes) { fprintf(stderr, "[vkrun] MAXW 耗尽\n"); return -1; }
    fprintf(stderr, "[vkrun] W 分成 %d 块:", vg->n_w);
    for (int i = 0; i < vg->n_w; i++)
        fprintf(stderr, " [%d]=%.2fGB(%s)", i, vg->wbytes[i] / 1073741824.0, vg->wheap[i] ? "devlocal" : "gtt");
    fprintf(stderr, "\n");

    vg->xbytes = x_bytes; vg->ybytes = y_bytes; vg->sbytes = stage_bytes;
    if (mk_buf(vg, &vg->xb, &vg->xm, &vg->xmap, x_bytes, 0, type_gtt)) return -1;
    if (mk_buf(vg, &vg->yb, &vg->ym, &vg->ymap, y_bytes, 0, type_gtt)) return -1;
    if (mk_buf(vg, &vg->gb, &vg->gm, &vg->gmap, 8192, 0, type_gtt)) return -1;   // IQ2S 码表 4KB + IQ3S 网格 2KB
    if (mk_buf(vg, &vg->sb, &vg->sm, &vg->smap, stage_bytes, 0, type_gtt)) return -1;

    // ── shader/pipeline ──
    FILE* sp = fopen(spv_path, "rb");
    if (!sp) { fprintf(stderr, "[vkrun] 缺 %s\n", spv_path); return -1; }
    fseek(sp, 0, SEEK_END); long spsz = ftell(sp); rewind(sp);
    char* code = (char*)malloc((size_t)spsz);
    if (fread(code, 1, (size_t)spsz, sp) != (size_t)spsz) return -1;
    fclose(sp);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = (size_t)spsz, .pCode = (uint32_t*)code };
    VkShaderModule smod; CHECK(vkCreateShaderModule(vg->dev, &smci, NULL, &smod), "sm");
    VkDescriptorPoolSize dpsz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * MAXW };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = MAXW, .poolSizeCount = 1, .pPoolSizes = &dpsz };
    CHECK(vkCreateDescriptorPool(vg->dev, &dpci, NULL, &vg->dpool), "dpool");
    VkDescriptorSetLayoutBinding lb[4] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = 4, .pBindings = lb };
    CHECK(vkCreateDescriptorSetLayout(vg->dev, &dlci, NULL, &vg->dsl), "dsl");
    for (int i = 0; i < vg->n_w; i++) {
        VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = vg->dpool, .descriptorSetCount = 1,
                                             .pSetLayouts = &vg->dsl };
        CHECK(vkAllocateDescriptorSets(vg->dev, &dsai, &vg->dsets[i]), "dset");
        VkDescriptorBufferInfo dbi[4] = {
            { vg->wb[i], 0, VK_WHOLE_SIZE }, { vg->gb, 0, VK_WHOLE_SIZE },
            { vg->xb, 0, VK_WHOLE_SIZE }, { vg->yb, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet wr[4];
        for (int j = 0; j < 4; j++)
            wr[j] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                            .dstSet = vg->dsets[i], .dstBinding = (uint32_t)j,
                                            .descriptorCount = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .pBufferInfo = &dbi[j] };
        vkUpdateDescriptorSets(vg->dev, 4, wr, 0, NULL);
    }
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 20 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &vg->dsl,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    CHECK(vkCreatePipelineLayout(vg->dev, &plci, NULL, &vg->playout), "pl");
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                           .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = smod, .pName = "main" };
    VkComputePipelineCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                        .stage = ss, .layout = vg->playout };
    CHECK(vkCreateComputePipelines(vg->dev, NULL, 1, &pci, NULL, &vg->pipe), "pipe");
    vg->has_pipe2 = 0;
    if (spv2_path && spv2_path[0]) {
        FILE* sp2 = fopen(spv2_path, "rb");
        if (!sp2) { fprintf(stderr, "[vkrun] 缺 %s\n", spv2_path); return -1; }
        fseek(sp2, 0, SEEK_END); long sp2sz = ftell(sp2); rewind(sp2);
        char* code2 = (char*)malloc((size_t)sp2sz);
        if (fread(code2, 1, (size_t)sp2sz, sp2) != (size_t)sp2sz) return -1;
        fclose(sp2);
        VkShaderModuleCreateInfo smci2 = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                           .codeSize = (size_t)sp2sz, .pCode = (uint32_t*)code2 };
        VkShaderModule smod2; CHECK(vkCreateShaderModule(vg->dev, &smci2, NULL, &smod2), "sm2");
        VkPipelineShaderStageCreateInfo ss2 = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = smod2, .pName = "main" };
        VkComputePipelineCreateInfo pci2 = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                             .stage = ss2, .layout = vg->playout };
        CHECK(vkCreateComputePipelines(vg->dev, NULL, 1, &pci2, NULL, &vg->pipe2), "pipe2");
        vg->has_pipe2 = 1;
    }
    vg->has_pipe3 = 0;
    if (spv3_path && spv3_path[0]) {
        FILE* sp3 = fopen(spv3_path, "rb");
        if (!sp3) { fprintf(stderr, "[vkrun] 缺 %s\n", spv3_path); return -1; }
        fseek(sp3, 0, SEEK_END); long sp3sz = ftell(sp3); rewind(sp3);
        char* code3 = (char*)malloc((size_t)sp3sz);
        if (fread(code3, 1, (size_t)sp3sz, sp3) != (size_t)sp3sz) return -1;
        fclose(sp3);
        VkShaderModuleCreateInfo smci3 = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                           .codeSize = (size_t)sp3sz, .pCode = (uint32_t*)code3 };
        VkShaderModule smod3; CHECK(vkCreateShaderModule(vg->dev, &smci3, NULL, &smod3), "sm3");
        VkPipelineShaderStageCreateInfo ss3 = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = smod3, .pName = "main" };
        VkComputePipelineCreateInfo pci3 = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                             .stage = ss3, .layout = vg->playout };
        CHECK(vkCreateComputePipelines(vg->dev, NULL, 1, &pci3, NULL, &vg->pipe3), "pipe3");
        vg->has_pipe3 = 1;
    }
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = vg->qf };
    CHECK(vkCreateCommandPool(vg->dev, &cpci, NULL, &vg->cpool), "cpool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = vg->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    CHECK(vkAllocateCommandBuffers(vg->dev, &cbai, &vg->cmd), "cmd");
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    CHECK(vkCreateFence(vg->dev, &fci, NULL, &vg->fence), "fence");
    vg->wg_rows = getenv("VKRUN_WG_ROWS") ? (unsigned)atoi(getenv("VKRUN_WG_ROWS")) : 1u;
    pthread_mutex_init(&vg->qlock, NULL);
    VkCommandBufferAllocateInfo kbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = vg->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    CHECK(vkAllocateCommandBuffers(vg->dev, &kbai, &vg->kcmd), "kcmd");
    CHECK(vkCreateFence(vg->dev, &fci, NULL, &vg->kfence), "kfence");
    vg->keeper_on = (getenv("VKRUN_KEEPER") && strcmp(getenv("VKRUN_KEEPER"), "0") == 0) ? 0 : 1;
    vg->keeper_started = 0;
    if (vg->keeper_on && pthread_create(&vg->keeper_thr, NULL, keeper_fn, vg) == 0)
        vg->keeper_started = 1;
    else
        vg->keeper_on = 0;
    if (vg->keeper_on)
        fprintf(stderr, "[vkrun] keeper 线程已启动（DVFS 保活）\n");

    if (info) {
        info->n_w = vg->n_w;
        for (int i = 0; i < vg->n_w; i++) {
            info->w_bytes[i] = vg->wbytes[i];
            info->w_heap[i] = vg->wheap[i];
            info->w_map[i] = vg->wmap[i];
        }
    }
    *out = vg;
    return 0;
}

int vg_upload_grid(struct VG* vg, const void* src, unsigned long bytes /* ≤8192 */) {
    if (bytes > 8192) return -1;
    memcpy(vg->gmap, src, bytes);
    return 0;
}

static int submit_copy(VG* vg, unsigned long src_off, int wbuf, unsigned long dst_off, unsigned long len) {
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    pthread_mutex_lock(&vg->qlock);
    CHECK2(vkResetCommandBuffer(vg->cmd, 0), "rc");
    CHECK2(vkBeginCommandBuffer(vg->cmd, &bbi), "begin");
    VkBufferCopy bc = { src_off, dst_off, len };
    vkCmdCopyBuffer(vg->cmd, vg->sb, vg->wb[wbuf], 1, &bc);
    CHECK2(vkEndCommandBuffer(vg->cmd), "end");
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &vg->cmd };
    CHECK2(vkResetFences(vg->dev, 1, &vg->fence), "rf");
    CHECK2(vkQueueSubmit(vg->queue, 1, &si, vg->fence), "submit");
    CHECK2(vkWaitForFences(vg->dev, 1, &vg->fence, VK_TRUE, UINT64_MAX), "wait");
    pthread_mutex_unlock(&vg->qlock);
    return 0;
}

int vg_upload(struct VG* vg, int wbuf, unsigned long off, const void* src, unsigned long len) {
    if (wbuf < 0 || wbuf >= vg->n_w) return -1;
    if (off + len > vg->wbytes[wbuf]) return -1;
    if (!vg->wheap[wbuf]) {                    // GTT：直接 memcpy（CPU 写 GPU 可读）
        memcpy((char*)vg->wmap[wbuf] + off, src, len);
        return 0;
    }
    // DEV_LOCAL：staging 分块拷贝（OMP 并行 memcpy + 逐块 submit）
    const unsigned char* p = (const unsigned char*)src;
    unsigned long done = 0;
    double t0 = now_s();
    while (done < len) {
        unsigned long n = len - done;
        if (n > vg->sbytes) n = vg->sbytes;
#ifdef _OPENMP
        {   // 64B 条带并行拷贝
            long stripes = (long)((n + 63) / 64);
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < stripes; i++) {
                size_t o = (size_t)i * 64;
                size_t m = (n - o) > 64 ? 64 : (n - o);
                memcpy((char*)vg->smap + o, p + done + o, m);
            }
        }
#else
        memcpy(vg->smap, p + done, n);
#endif
        int rc = submit_copy(vg, 0, wbuf, off + done, n);
        if (rc != 0) return rc;
        done += n;
    }
    if (getenv("VKRUN_DBG"))
        fprintf(stderr, "[vkrun] upload buf%d+%lu %luMB %.1fs\n", wbuf, off,
                len >> 20, now_s() - t0);
    return 0;
}

int vg_run(struct VG* vg, const struct VGMat* mats, int n, const float* x, unsigned long x_floats) {
    double tt0 = getenv("VKRUN_TIMING") ? now_s() : 0;
    if (x_floats * 4 > vg->xbytes) return -1;
    if (x) memcpy(vg->xmap, x, x_floats * 4);   // x=NULL：调用者已直接写 X 映射
    for (int i = 0; i < n; i++)
        if (mats[i].wset >= (uint32_t)vg->n_w) return -2;
    pthread_mutex_lock(&vg->qlock);
    CHECK2(vkResetCommandBuffer(vg->cmd, 0), "rc");
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    CHECK2(vkBeginCommandBuffer(vg->cmd, &bbi), "begin");
    int cur_set = -1, cur_pipe = -1;
    for (int i = 0; i < n; i++) {
        if (mats[i].wset != (uint32_t)cur_set) {
            vkCmdBindDescriptorSets(vg->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vg->playout,
                                    0, 1, &vg->dsets[mats[i].wset], 0, NULL);
            cur_set = (int)mats[i].wset;
        }
        uint32_t p = 0;
        if (mats[i].pipe == 1 && vg->has_pipe2) p = 1;
        if (mats[i].pipe == 2 && vg->has_pipe3) p = 2;
        if (p != (uint32_t)cur_pipe) {
            vkCmdBindPipeline(vg->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              p == 0 ? vg->pipe : (p == 1 ? vg->pipe2 : vg->pipe3));
            cur_pipe = (int)p;
        }
        uint32_t pc[7] = { mats[i].w_off, mats[i].x_off, mats[i].y_off, mats[i].n_out, mats[i].nb, 0, 0 };
        vkCmdPushConstants(vg->cmd, vg->playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, pc);
        uint32_t wgr = (p == 1u) ? 4u : (vg->wg_rows > 0 ? vg->wg_rows : 1u);
        // pipe2(GEMM16) 与 pipe0 同为 8 行/WG ⇒ 用 wg_rows
        uint32_t groups = (mats[i].n_out + wgr - 1u) / wgr;
        vkCmdDispatch(vg->cmd, groups, 1, 1);
    }
    CHECK2(vkEndCommandBuffer(vg->cmd), "end");
    double t_rec = getenv("VKRUN_TIMING") ? now_s() - tt0 : 0;
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &vg->cmd };
    CHECK2(vkResetFences(vg->dev, 1, &vg->fence), "rf");
    CHECK2(vkQueueSubmit(vg->queue, 1, &si, vg->fence), "submit");
    double t_sub = getenv("VKRUN_TIMING") ? now_s() - tt0 - t_rec : 0;
    CHECK2(vkWaitForFences(vg->dev, 1, &vg->fence, VK_TRUE, UINT64_MAX), "wait");
    pthread_mutex_unlock(&vg->qlock);
    if (getenv("VKRUN_TIMING"))
        fprintf(stderr, "[vgt] n=%d record=%.3f submit=%.3f wait=%.3f ms\n",
                n, t_rec * 1e3, t_sub * 1e3, (now_s() - tt0 - t_rec - t_sub) * 1e3);
    return 0;
}

void* vg_xmap(struct VG* vg) { return vg->xmap; }
void* vg_ymap(struct VG* vg) { return vg->ymap; }
void vg_free(struct VG* vg) {
    if (!vg) return;
    if (vg->keeper_started) {
        vg->keeper_on = 0;
        pthread_mutex_lock(&vg->qlock); pthread_mutex_unlock(&vg->qlock);  // 确保离开临界区
        pthread_join(vg->keeper_thr, NULL);
    }
    vkDeviceWaitIdle(vg->dev);
    for (int i = 0; i < vg->n_w; i++) { vkDestroyBuffer(vg->dev, vg->wb[i], NULL); vkFreeMemory(vg->dev, vg->wm[i], NULL); }
    vkDestroyBuffer(vg->dev, vg->xb, NULL); vkFreeMemory(vg->dev, vg->xm, NULL);
    vkDestroyBuffer(vg->dev, vg->yb, NULL); vkFreeMemory(vg->dev, vg->ym, NULL);
    vkDestroyBuffer(vg->dev, vg->gb, NULL); vkFreeMemory(vg->dev, vg->gm, NULL);
    vkDestroyBuffer(vg->dev, vg->sb, NULL); vkFreeMemory(vg->dev, vg->sm, NULL);
    vkDestroyCommandPool(vg->dev, vg->cpool, NULL);
    vkDestroyPipeline(vg->dev, vg->pipe, NULL);
    vkDestroyPipelineLayout(vg->dev, vg->playout, NULL);
    vkDestroyDescriptorPool(vg->dev, vg->dpool, NULL);
    vkDestroyDescriptorSetLayout(vg->dev, vg->dsl, NULL);
    vkDestroyDevice(vg->dev, NULL);
    vkDestroyInstance(vg->inst, NULL);
    free(vg);
}
