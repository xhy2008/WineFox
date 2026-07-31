#include "vulkan_app.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef NDEBUG
#define WF_VK_VALIDATION 1
#else
#define WF_VK_VALIDATION 0
#endif

namespace vkt {

// ===========================================================================
// Room geometry — 20×20×20 enclosed box, viewed from INSIDE.
// 24 vertices (4 per face, distinct normals) + 36 indices.
// Normals point INWARD (toward the camera). Each wall has a distinct color.
// ===========================================================================

static const std::vector<Vertex> kRoomVertices = {
    // +X wall (right, red) — normal points -X (inward)
    {{ 10.0f, -10.0f,  10.0f}, {0.8f, 0.2f, 0.2f}, {-1.0f, 0.0f, 0.0f}},
    {{ 10.0f, -10.0f, -10.0f}, {0.8f, 0.2f, 0.2f}, {-1.0f, 0.0f, 0.0f}},
    {{ 10.0f,  10.0f, -10.0f}, {0.8f, 0.2f, 0.2f}, {-1.0f, 0.0f, 0.0f}},
    {{ 10.0f,  10.0f,  10.0f}, {0.8f, 0.2f, 0.2f}, {-1.0f, 0.0f, 0.0f}},
    // -X wall (left, green) — normal points +X (inward)
    {{-10.0f, -10.0f, -10.0f}, {0.2f, 0.8f, 0.2f}, {1.0f, 0.0f, 0.0f}},
    {{-10.0f, -10.0f,  10.0f}, {0.2f, 0.8f, 0.2f}, {1.0f, 0.0f, 0.0f}},
    {{-10.0f,  10.0f,  10.0f}, {0.2f, 0.8f, 0.2f}, {1.0f, 0.0f, 0.0f}},
    {{-10.0f,  10.0f, -10.0f}, {0.2f, 0.8f, 0.2f}, {1.0f, 0.0f, 0.0f}},
    // +Y wall (ceiling, blue) — normal points -Y (inward)
    {{-10.0f,  10.0f,  10.0f}, {0.2f, 0.2f, 0.8f}, {0.0f, -1.0f, 0.0f}},
    {{ 10.0f,  10.0f,  10.0f}, {0.2f, 0.2f, 0.8f}, {0.0f, -1.0f, 0.0f}},
    {{ 10.0f,  10.0f, -10.0f}, {0.2f, 0.2f, 0.8f}, {0.0f, -1.0f, 0.0f}},
    {{-10.0f,  10.0f, -10.0f}, {0.2f, 0.2f, 0.8f}, {0.0f, -1.0f, 0.0f}},
    // -Y wall (floor, yellow) — normal points +Y (inward)
    {{-10.0f, -10.0f, -10.0f}, {0.8f, 0.8f, 0.2f}, {0.0f, 1.0f, 0.0f}},
    {{ 10.0f, -10.0f, -10.0f}, {0.8f, 0.8f, 0.2f}, {0.0f, 1.0f, 0.0f}},
    {{ 10.0f, -10.0f,  10.0f}, {0.8f, 0.8f, 0.2f}, {0.0f, 1.0f, 0.0f}},
    {{-10.0f, -10.0f,  10.0f}, {0.8f, 0.8f, 0.2f}, {0.0f, 1.0f, 0.0f}},
    // +Z wall (front, cyan) — normal points -Z (inward)
    {{-10.0f, -10.0f,  10.0f}, {0.2f, 0.8f, 0.8f}, {0.0f, 0.0f, -1.0f}},
    {{ 10.0f, -10.0f,  10.0f}, {0.2f, 0.8f, 0.8f}, {0.0f, 0.0f, -1.0f}},
    {{ 10.0f,  10.0f,  10.0f}, {0.2f, 0.8f, 0.8f}, {0.0f, 0.0f, -1.0f}},
    {{-10.0f,  10.0f,  10.0f}, {0.2f, 0.8f, 0.8f}, {0.0f, 0.0f, -1.0f}},
    // -Z wall (back, magenta) — normal points +Z (inward)
    {{ 10.0f, -10.0f, -10.0f}, {0.8f, 0.2f, 0.8f}, {0.0f, 0.0f, 1.0f}},
    {{-10.0f, -10.0f, -10.0f}, {0.8f, 0.2f, 0.8f}, {0.0f, 0.0f, 1.0f}},
    {{-10.0f,  10.0f, -10.0f}, {0.8f, 0.2f, 0.8f}, {0.0f, 0.0f, 1.0f}},
    {{ 10.0f,  10.0f, -10.0f}, {0.8f, 0.2f, 0.8f}, {0.0f, 0.0f, 1.0f}},
};

// Indices are wound so that each face is FRONT-facing (CCW) when viewed
// from INSIDE the room (i.e. from the side the inward normal points to).
// This lets us enable back-face culling and still see all walls.
static const std::vector<uint16_t> kRoomIndices = {
     2,  1,  0,   3,  2,  0,  // +X (reversed)
     6,  5,  4,   7,  6,  4,  // -X (reversed)
    10,  9,  8,  11, 10,  8,  // +Y (reversed)
    14, 13, 12,  15, 14, 12,  // -Y (reversed)
    18, 17, 16,  19, 18, 16,  // +Z (reversed)
    22, 21, 20,  23, 22, 20,  // -Z (reversed)
};

// ===========================================================================
// Anonymous-namespace helpers
// ===========================================================================

namespace {

#if WF_VK_VALIDATION
const char* kValidationLayers[] = {"VK_LAYER_KHRONOS_validation"};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        std::fprintf(stderr, "[vk] ERROR: %s\n", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[vk] WARN: %s\n", data->pMessage);
    }
    return VK_FALSE;
}
#endif

uint32_t find_graphics_family(VkPhysicalDevice dev, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        VkBool32 present = VK_FALSE;
        if (surface) vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present);
        if (surface && !present) continue;
        return i;
    }
    return UINT32_MAX;
}

VkSurfaceFormatKHR pick_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats[0];
}

VkPresentModeKHR pick_present_mode(const std::vector<VkPresentModeKHR>& modes) {
    // Force V-Sync via FIFO: present only on vertical retrace, no tearing.
    // (FIFO is always supported per the Vulkan spec.)
    (void)modes;
    return VK_PRESENT_MODE_FIFO_KHR;
}

} // namespace

// ===========================================================================
// Init
// ===========================================================================

bool VulkanApp::init(SDL_Window* window, int width, int height) {
    if (!create_instance_())   { std::fprintf(stderr, "[vkt] FAILED: create_instance_\n");   return false; }
    if (!setup_debug_messenger_()) { std::fprintf(stderr, "[vkt] FAILED: setup_debug_messenger_\n"); return false; }
    if (!create_surface_(window)) { std::fprintf(stderr, "[vkt] FAILED: create_surface_\n"); return false; }
    if (!pick_physical_device_()) { std::fprintf(stderr, "[vkt] FAILED: pick_physical_device_\n"); return false; }
    if (!create_device_())     { std::fprintf(stderr, "[vkt] FAILED: create_device_\n");     return false; }
    if (!create_swapchain_(width, height)) { std::fprintf(stderr, "[vkt] FAILED: create_swapchain_\n"); return false; }
    if (!create_image_views_()) { std::fprintf(stderr, "[vkt] FAILED: create_image_views_\n"); return false; }
    if (!find_depth_format_())  { std::fprintf(stderr, "[vkt] FAILED: find_depth_format_\n");  return false; }
    if (!create_depth_resources_()) { std::fprintf(stderr, "[vkt] FAILED: create_depth_resources_\n"); return false; }
    if (!create_render_pass_()) { std::fprintf(stderr, "[vkt] FAILED: create_render_pass_\n"); return false; }
    if (!create_pipeline_())    { std::fprintf(stderr, "[vkt] FAILED: create_pipeline_\n");    return false; }
    if (!create_framebuffers_()) { std::fprintf(stderr, "[vkt] FAILED: create_framebuffers_\n"); return false; }
    if (!create_command_pool_()) { std::fprintf(stderr, "[vkt] FAILED: create_command_pool_\n"); return false; }
    if (!create_vertex_buffer_()) { std::fprintf(stderr, "[vkt] FAILED: create_vertex_buffer_\n"); return false; }
    if (!create_index_buffer_())  { std::fprintf(stderr, "[vkt] FAILED: create_index_buffer_\n");  return false; }
    if (!create_command_buffers_()) { std::fprintf(stderr, "[vkt] FAILED: create_command_buffers_\n"); return false; }
    if (!create_sync_objects_())  { std::fprintf(stderr, "[vkt] FAILED: create_sync_objects_\n");  return false; }

    camera_.set_perspective(45.0f, (float)width / (float)height, 0.1f, 100.0f);
    // Start at room center, looking toward -Z (toward the back/magenta wall).
    camera_.set_position(glm::vec3(0.0f, 0.0f, 0.0f));
    std::fprintf(stderr, "[vkt] Vulkan ready (extent=%ux%u, depth=%d)\n",
                 swapchain_extent_.width, swapchain_extent_.height,
                 static_cast<int>(depth_format_));
    return true;
}

bool VulkanApp::create_instance_() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vulkan-test";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "vulkan-test";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    uint32_t sdl_ext_count = 0;
    const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);
    std::vector<const char*> exts(sdl_exts, sdl_exts + sdl_ext_count);

    VkDebugUtilsMessengerCreateInfoEXT debug_ci{};
#if WF_VK_VALIDATION
    exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    debug_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_ci.pfnUserCallback = debug_callback;
    ci.pNext = &debug_ci;
    ci.enabledLayerCount = 1;
    ci.ppEnabledLayerNames = kValidationLayers;
#endif

    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();

    if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateInstance failed\n");
        return false;
    }
    return true;
}

bool VulkanApp::setup_debug_messenger_() {
#if WF_VK_VALIDATION
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) return true; // extension not available — non-fatal

    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debug_callback;

    if (fn(instance_, &ci, nullptr, &debug_messenger_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] failed to create debug messenger\n");
    }
#endif
    return true;
}

bool VulkanApp::create_surface_(SDL_Window* window) {
    if (!SDL_Vulkan_CreateSurface(window, instance_, nullptr, &surface_)) {
        std::fprintf(stderr, "[vkt] SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        return false;
    }
    return true;
}

bool VulkanApp::pick_physical_device_() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        std::fprintf(stderr, "[vkt] no GPU with Vulkan support\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    for (VkPhysicalDevice dev : devices) {
        uint32_t family = find_graphics_family(dev, surface_);
        if (family == UINT32_MAX) continue;

        // Check swapchain extension.
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, exts.data());
        bool has_swapchain = false;
        for (const auto& e : exts) {
            if (strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                has_swapchain = true; break;
            }
        }
        if (!has_swapchain) continue;

        physical_device_ = dev;
        graphics_family_ = family;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        std::fprintf(stderr, "[vkt] GPU: %s\n", props.deviceName);
        return true;
    }

    std::fprintf(stderr, "[vkt] no suitable GPU found\n");
    return false;
}

bool VulkanApp::create_device_() {
    std::fprintf(stderr, "[vkt] create_device_ enter (family=%u)\n", graphics_family_);
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = graphics_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_priority;

    const char* dev_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pQueueCreateInfos = &qci;
    ci.queueCreateInfoCount = 1;
    ci.ppEnabledExtensionNames = dev_exts;
    ci.enabledExtensionCount = 1;
    ci.pEnabledFeatures = &features;

    VkResult res = vkCreateDevice(physical_device_, &ci, nullptr, &device_);
    std::fprintf(stderr, "[vkt] vkCreateDevice = %d\n", res);
    if (res != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
    std::fprintf(stderr, "[vkt] create_device_ done\n");
    return true;
}

bool VulkanApp::create_swapchain_(int width, int height) {
    std::fprintf(stderr, "[vkt] create_swapchain_ enter (%dx%d)\n", width, height);
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);
    std::fprintf(stderr, "[vkt]   caps: %ux%u, minImg=%u, maxImg=%u\n",
                 caps.currentExtent.width, caps.currentExtent.height,
                 caps.minImageCount, caps.maxImageCount);

    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &fmt_count, nullptr);
    std::fprintf(stderr, "[vkt]   fmt_count=%u\n", fmt_count);
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &fmt_count, formats.data());

    uint32_t mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, nullptr);
    std::fprintf(stderr, "[vkt]   mode_count=%u\n", mode_count);
    std::vector<VkPresentModeKHR> modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, modes.data());

    VkSurfaceFormatKHR sf = pick_surface_format(formats);
    VkPresentModeKHR pm = pick_present_mode(modes);
    std::fprintf(stderr, "[vkt]   format=%d, present=%d\n", sf.format, pm);

    VkExtent2D extent;
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width = std::max(caps.minImageExtent.width,
                                std::min(caps.maxImageExtent.width, (uint32_t)width));
        extent.height = std::max(caps.minImageExtent.height,
                                 std::min(caps.maxImageExtent.height, (uint32_t)height));
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = image_count;
    ci.imageFormat = sf.format;
    ci.imageColorSpace = sf.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    std::fprintf(stderr, "[vkt]   vkCreateSwapchainKHR (extent=%ux%u, imgCount=%u)...\n",
                 extent.width, extent.height, image_count);
    VkResult sc_res = vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_);
    std::fprintf(stderr, "[vkt]   vkCreateSwapchainKHR = %d\n", sc_res);
    if (sc_res != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateSwapchainKHR failed\n");
        return false;
    }

    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    std::fprintf(stderr, "[vkt]   swapchain images=%u\n", image_count);
    swapchain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());

    swapchain_format_ = sf.format;
    swapchain_extent_ = extent;
    std::fprintf(stderr, "[vkt] create_swapchain_ done\n");
    return true;
}

bool VulkanApp::create_image_views_() {
    std::fprintf(stderr, "[vkt] create_image_views_ (count=%zu)\n", swapchain_images_.size());
    swapchain_image_views_.resize(swapchain_images_.size());
    for (size_t i = 0; i < swapchain_images_.size(); ++i) {
        std::fprintf(stderr, "[vkt]   image_view[%zu] image=%p\n", i, (void*)swapchain_images_[i]);
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = swapchain_images_[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = swapchain_format_;
        ci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = 1;

        VkResult iv_res = vkCreateImageView(device_, &ci, nullptr, &swapchain_image_views_[i]);
        std::fprintf(stderr, "[vkt]   image_view[%zu] result=%d\n", i, iv_res);
        if (iv_res != VK_SUCCESS) {
            std::fprintf(stderr, "[vkt] vkCreateImageView failed\n");
            return false;
        }
    }
    std::fprintf(stderr, "[vkt] create_image_views_ done\n");
    return true;
}

// ===========================================================================
// Depth buffer
// ===========================================================================

bool VulkanApp::find_supported_format_(const std::vector<VkFormat>& candidates,
                                       VkImageTiling tiling,
                                       VkFormatFeatureFlags features,
                                       VkFormat& out) const {
    for (VkFormat f : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physical_device_, f, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features) {
            out = f; return true;
        }
        if (tiling == VK_IMAGE_TILING_OPTIMAL &&
            (props.optimalTilingFeatures & features) == features) {
            out = f; return true;
        }
    }
    return false;
}

bool VulkanApp::find_depth_format_() {
    bool ok = find_supported_format_(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        depth_format_);
    std::fprintf(stderr, "[vkt] find_depth_format_ = %d (format=%d)\n", ok, (int)depth_format_);
    return ok;
}

bool VulkanApp::create_depth_resources_() {
    std::fprintf(stderr, "[vkt] create_depth_resources_ enter\n");
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = depth_format_;
    ici.extent = {swapchain_extent_.width, swapchain_extent_.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    std::fprintf(stderr, "[vkt]   vkCreateImage...\n");
    if (vkCreateImage(device_, &ici, nullptr, &depth_image_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateImage (depth) failed\n");
        return false;
    }
    std::fprintf(stderr, "[vkt]   vkGetImageMemoryRequirements...\n");
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, depth_image_, &req);
    std::fprintf(stderr, "[vkt]   req.size=%llu, typeBits=0x%x\n",
                 (unsigned long long)req.size, req.memoryTypeBits);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type_(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::fprintf(stderr, "[vkt]   memType=%u\n", alloc.memoryTypeIndex);

    std::fprintf(stderr, "[vkt]   vkAllocateMemory...\n");
    if (vkAllocateMemory(device_, &alloc, nullptr, &depth_image_memory_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkAllocateMemory (depth) failed\n");
        return false;
    }
    std::fprintf(stderr, "[vkt]   vkBindImageMemory...\n");
    vkBindImageMemory(device_, depth_image_, depth_image_memory_, 0);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = depth_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = depth_format_;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;

    std::fprintf(stderr, "[vkt]   vkCreateImageView (depth)...\n");
    if (vkCreateImageView(device_, &vci, nullptr, &depth_image_view_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateImageView (depth) failed\n");
        return false;
    }
    std::fprintf(stderr, "[vkt] create_depth_resources_ done\n");
    return true;
}

// ===========================================================================
// Render pass + graphics pipeline
// ===========================================================================

bool VulkanApp::create_render_pass_() {
    VkAttachmentDescription color_attach{};
    color_attach.format = swapchain_format_;
    color_attach.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth_attach{};
    depth_attach.format = depth_format_;
    depth_attach.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attach.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attach.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_ref{};
    depth_ref.attachment = 1;
    depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array attachments = {color_attach, depth_attach};
    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
    ci.pAttachments = attachments.data();
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;

    VkResult rp_res = vkCreateRenderPass(device_, &ci, nullptr, &render_pass_);
    std::fprintf(stderr, "[vkt] create_render_pass_ = %d\n", rp_res);
    return rp_res == VK_SUCCESS;
}

bool VulkanApp::create_pipeline_() {
    std::fprintf(stderr, "[vkt] create_pipeline_ enter\n");
    VkShaderModule vert = load_shader_("shaders/cube.vert.spv");
    VkShaderModule frag = load_shader_("shaders/cube.frag.spv");
    // Note: glslc outputs cube.vert.spv and cube.frag.spv (full name + .spv)
    std::fprintf(stderr, "[vkt]   shaders loaded: vert=%p, frag=%p\n", (void*)vert, (void*)frag);
    if (!vert || !frag) return false;

    // IMPORTANT: = {} zero-initializes pNext and flags. Without it,
    // vkCreateGraphicsPipelines dereferences the garbage pNext pointer
    // and crashes with an access violation.
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    // Vertex input: pos(3f) + color(3f) + normal(3f) = 36 bytes
    std::array<VkVertexInputBindingDescription, 1> bind_desc{};
    bind_desc[0].binding = 0;
    bind_desc[0].stride = sizeof(Vertex);
    bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> attr_desc{};
    attr_desc[0].location = 0; // pos
    attr_desc[0].binding = 0;
    attr_desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attr_desc[0].offset = offsetof(Vertex, pos);
    attr_desc[1].location = 1; // color
    attr_desc[1].binding = 0;
    attr_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attr_desc[1].offset = offsetof(Vertex, color);
    attr_desc[2].location = 2; // normal
    attr_desc[2].binding = 0;
    attr_desc[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attr_desc[2].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = bind_desc.data();
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = attr_desc.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    ia.primitiveRestartEnable = VK_FALSE;

    // Viewport + scissor are dynamic (set in command buffer).
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // Back-face culling enabled. Room indices are wound CCW from inside so
    // walls are front-facing when viewed from the interior. Essential for
    // performance when rendering complex models later.
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend_attach{};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.logicOpEnable = VK_FALSE;
    cb.attachmentCount = 1;
    cb.pAttachments = &blend_attach;

    std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn_state{};
    dyn_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_state.dynamicStateCount = 2;
    dyn_state.pDynamicStates = dyn.data();

    // Push constant: single mat4 (64 bytes) for MVP.
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset = 0;
    pc.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;

    std::fprintf(stderr, "[vkt]   vkCreatePipelineLayout...\n");
    if (vkCreatePipelineLayout(device_, &plci, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreatePipelineLayout failed\n");
        return false;
    }
    std::fprintf(stderr, "[vkt]   pipeline_layout=%p\n", (void*)pipeline_layout_);

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dyn_state;
    pci.layout = pipeline_layout_;
    pci.renderPass = render_pass_;
    pci.subpass = 0;

    std::fprintf(stderr, "[vkt]   vkCreateGraphicsPipelines...\n");
    VkResult gp_res = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_);
    std::fprintf(stderr, "[vkt]   vkCreateGraphicsPipelines = %d, pipeline=%p\n", gp_res, (void*)pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    std::fprintf(stderr, "[vkt] create_pipeline_ done\n");
    return gp_res == VK_SUCCESS;
}

bool VulkanApp::create_framebuffers_() {
    framebuffers_.resize(swapchain_image_views_.size());
    for (size_t i = 0; i < swapchain_image_views_.size(); ++i) {
        std::array<VkImageView, 2> attachments = {
            swapchain_image_views_[i],
            depth_image_view_
        };

        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = render_pass_;
        ci.attachmentCount = static_cast<uint32_t>(attachments.size());
        ci.pAttachments = attachments.data();
        ci.width = swapchain_extent_.width;
        ci.height = swapchain_extent_.height;
        ci.layers = 1;

        if (vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
            std::fprintf(stderr, "[vkt] vkCreateFramebuffer failed\n");
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Command pool + buffers
// ===========================================================================

bool VulkanApp::create_command_pool_() {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphics_family_;

    if (vkCreateCommandPool(device_, &ci, nullptr, &command_pool_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateCommandPool failed\n");
        return false;
    }
    return true;
}

bool VulkanApp::create_command_buffers_() {
    command_buffers_.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = command_pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    return vkAllocateCommandBuffers(device_, &alloc, command_buffers_.data()) == VK_SUCCESS;
}

// ===========================================================================
// Vertex + index buffers (with staging)
// ===========================================================================

void VulkanApp::create_buffer_(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags props,
                               VkBuffer& buffer, VkDeviceMemory& memory) const {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bci, nullptr, &buffer) != VK_SUCCESS) return;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type_(req.memoryTypeBits, props);

    if (vkAllocateMemory(device_, &alloc, nullptr, &memory) != VK_SUCCESS) return;
    vkBindBufferMemory(device_, buffer, memory, 0);
}

void VulkanApp::copy_buffer_(VkBuffer src, VkBuffer dst, VkDeviceSize size) const {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = command_pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    vkQueueSubmit(graphics_queue_, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

bool VulkanApp::create_vertex_buffer_() {
    VkDeviceSize size = sizeof(kRoomVertices[0]) * kRoomVertices.size();

    VkBuffer staging;
    VkDeviceMemory staging_mem;
    create_buffer_(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, staging_mem);

    void* data = nullptr;
    vkMapMemory(device_, staging_mem, 0, size, 0, &data);
    memcpy(data, kRoomVertices.data(), (size_t)size);
    vkUnmapMemory(device_, staging_mem);

    create_buffer_(size,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   vertex_buffer_, vertex_buffer_memory_);

    copy_buffer_(staging, vertex_buffer_, size);

    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, staging_mem, nullptr);
    return vertex_buffer_ != VK_NULL_HANDLE;
}

bool VulkanApp::create_index_buffer_() {
    index_count_ = static_cast<uint32_t>(kRoomIndices.size());
    VkDeviceSize size = sizeof(kRoomIndices[0]) * kRoomIndices.size();

    VkBuffer staging;
    VkDeviceMemory staging_mem;
    create_buffer_(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, staging_mem);

    void* data = nullptr;
    vkMapMemory(device_, staging_mem, 0, size, 0, &data);
    memcpy(data, kRoomIndices.data(), (size_t)size);
    vkUnmapMemory(device_, staging_mem);

    create_buffer_(size,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   index_buffer_, index_buffer_memory_);

    copy_buffer_(staging, index_buffer_, size);

    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, staging_mem, nullptr);
    return index_buffer_ != VK_NULL_HANDLE;
}

// ===========================================================================
// Sync objects
// ===========================================================================

bool VulkanApp::create_sync_objects_() {
    image_available_.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(device_, &sci, nullptr, &image_available_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &sci, nullptr, &render_finished_[i]) != VK_SUCCESS ||
            vkCreateFence(device_, &fci, nullptr, &in_flight_[i]) != VK_SUCCESS) {
            std::fprintf(stderr, "[vkt] failed to create sync objects\n");
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Draw
// ===========================================================================

void VulkanApp::record_command_buffer_(VkCommandBuffer cmd, uint32_t image_index,
                                        const glm::mat4& mvp) {
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    std::array<VkClearValue, 2> clears{};
    clears[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = render_pass_;
    rpbi.framebuffer = framebuffers_[image_index];
    rpbi.renderArea.offset = {0, 0};
    rpbi.renderArea.extent = swapchain_extent_;
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues = clears.data();

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkViewport vp{};
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.width = (float)swapchain_extent_.width;
    vp.height = (float)swapchain_extent_.height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchain_extent_;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer vb = vertex_buffer_;
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
    vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT16);

    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(glm::mat4), &mvp);

    vkCmdDrawIndexed(cmd, index_count_, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void VulkanApp::draw_frame() {
    vkWaitForFences(device_, 1, &in_flight_[current_frame_], VK_TRUE, UINT64_MAX);

    VkResult res = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                         image_available_[current_frame_],
                                         VK_NULL_HANDLE, &image_index_);

    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain_(swapchain_extent_.width, swapchain_extent_.height);
        return;
    }

    vkResetFences(device_, 1, &in_flight_[current_frame_]);

    vkResetCommandBuffer(command_buffers_[current_frame_], 0);

    // Room is static — no model transform, just camera view-projection.
    glm::mat4 mvp = camera_.view_projection();

    record_command_buffer_(command_buffers_[current_frame_], image_index_, mvp);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_sem = image_available_[current_frame_];
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &wait_sem;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffers_[current_frame_];

    VkSemaphore signal_sem = render_finished_[current_frame_];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signal_sem;

    if (vkQueueSubmit(graphics_queue_, 1, &submit, in_flight_[current_frame_]) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkQueueSubmit failed\n");
        return;
    }

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &signal_sem;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &image_index_;

    res = vkQueuePresentKHR(graphics_queue_, &present);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || framebuffer_resized_) {
        framebuffer_resized_ = false;
        recreate_swapchain_(swapchain_extent_.width, swapchain_extent_.height);
    }

    current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ===========================================================================
// Swapchain recreation
// ===========================================================================

void VulkanApp::on_resize(int width, int height) {
    framebuffer_resized_ = true;
}

void VulkanApp::cleanup_swapchain_() {
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    framebuffers_.clear();

    vkDestroyImageView(device_, depth_image_view_, nullptr);
    vkDestroyImage(device_, depth_image_, nullptr);
    vkFreeMemory(device_, depth_image_memory_, nullptr);
    depth_image_ = VK_NULL_HANDLE;
    depth_image_view_ = VK_NULL_HANDLE;

    for (auto iv : swapchain_image_views_) vkDestroyImageView(device_, iv, nullptr);
    swapchain_image_views_.clear();

    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void VulkanApp::recreate_swapchain_(int width, int height) {
    vkDeviceWaitIdle(device_);

    cleanup_swapchain_();

    create_swapchain_(width, height);
    create_image_views_();
    find_depth_format_();
    create_depth_resources_();
    create_framebuffers_();

    camera_.set_aspect((float)swapchain_extent_.width / (float)swapchain_extent_.height);
}

// ===========================================================================
// Helpers
// ===========================================================================

uint32_t VulkanApp::find_memory_type_(uint32_t type_filter, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0;
}

VkShaderModule VulkanApp::load_shader_(const char* path) const {
    // Resolve path relative to the executable directory (not CWD).
    // On Windows, GetModuleFileNameA gives the full exe path.
    std::string full_path = path;
#ifdef _WIN32
    {
        char exe_path[MAX_PATH];
        DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
        if (len > 0) {
            std::string exe_dir(exe_path, len);
            size_t slash = exe_dir.find_last_of("\\/");
            if (slash != std::string::npos) {
                full_path = exe_dir.substr(0, slash + 1) + path;
            }
        }
    }
#endif

    std::ifstream file(full_path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[vkt] failed to open shader: %s (cwd-relative: %s)\n",
                     full_path.c_str(), path);
        return VK_NULL_HANDLE;
    }
    size_t size = (size_t)file.tellg();
    std::vector<char> code(size);
    file.seekg(0);
    file.read(code.data(), size);

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod;
    if (vkCreateShaderModule(device_, &ci, nullptr, &mod) != VK_SUCCESS) {
        std::fprintf(stderr, "[vkt] vkCreateShaderModule failed for %s\n", full_path.c_str());
        return VK_NULL_HANDLE;
    }
    return mod;
}

// ===========================================================================
// Destructor
// ===========================================================================

VulkanApp::~VulkanApp() {
    if (device_) vkDeviceWaitIdle(device_);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (in_flight_[i])       vkDestroyFence(device_, in_flight_[i], nullptr);
        if (render_finished_[i]) vkDestroySemaphore(device_, render_finished_[i], nullptr);
        if (image_available_[i]) vkDestroySemaphore(device_, image_available_[i], nullptr);
    }

    if (index_buffer_)  { vkDestroyBuffer(device_, index_buffer_, nullptr);  vkFreeMemory(device_, index_buffer_memory_, nullptr); }
    if (vertex_buffer_) { vkDestroyBuffer(device_, vertex_buffer_, nullptr); vkFreeMemory(device_, vertex_buffer_memory_, nullptr); }

    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);

    cleanup_swapchain_();

    if (pipeline_)        vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (render_pass_)     vkDestroyRenderPass(device_, render_pass_, nullptr);
    if (device_)          vkDestroyDevice(device_, nullptr);

#if WF_VK_VALIDATION
    if (debug_messenger_) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance_, debug_messenger_, nullptr);
    }
#endif

    if (surface_)  vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

} // namespace vkt
