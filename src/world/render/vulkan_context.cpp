#include "vulkan_context.h"

#include <cstdio>
#include <cstring>
#include <vector>

#ifndef NDEBUG
#define WF_VK_VALIDATION 1
#else
#define WF_VK_VALIDATION 0
#endif

namespace winefox {
namespace world {

namespace {

const char* kValidationLayers[] = {
    "VK_LAYER_KHRONOS_validation"
};

bool check_validation_support() {
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available.data());

    for (const char* name : kValidationLayers) {
        bool found = false;
        for (const auto& layer : available) {
            if (strcmp(layer.layerName, name) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* user_data)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        std::fprintf(stderr, "[vk] ERROR: %s\n", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[vk] WARN: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

uint32_t find_graphics_family(VkPhysicalDevice dev) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
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
    for (VkPresentModeKHR m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;  // low-latency
    }
    return VK_PRESENT_MODE_FIFO_KHR;  // always available, vsync
}

} // namespace

bool VulkanContext::init(SDL_Window* window, int width, int height) {
    if (!create_instance_()) return false;
    if (!create_surface_(window)) return false;
    if (!pick_physical_device_()) return false;
    if (!create_device_()) return false;
    if (!create_swapchain_(width, height)) return false;
    if (!create_image_views_()) return false;
    if (!create_command_pool_()) return false;
    if (!create_sync_objects_()) return false;

    std::fprintf(stderr, "[vk] Vulkan context ready (extent=%ux%u)\n",
                 swapchain_extent_.width, swapchain_extent_.height);
    return true;
}

bool VulkanContext::create_instance_() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "WineFox World";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "WineFox";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    // Extensions needed for SDL surface + debug utils (if validation enabled).
    uint32_t sdl_ext_count = 0;
    const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);

    std::vector<const char*> extensions(sdl_exts, sdl_exts + sdl_ext_count);

    VkDebugUtilsMessengerCreateInfoEXT debug_ci{};
#if WF_VK_VALIDATION
    bool use_validation = check_validation_support();
    if (use_validation) {
        // Must add VK_EXT_debug_utils to the extension list when using
        // VkDebugUtilsMessengerCreateInfoEXT in the pNext chain.
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        debug_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_ci.pfnUserCallback = debug_callback;
        create_info.pNext = &debug_ci;
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = kValidationLayers;
        std::fprintf(stderr, "[vk] validation layers enabled\n");
    } else {
        std::fprintf(stderr, "[vk] validation layers not available\n");
    }
#endif

    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    VkResult res = vkCreateInstance(&create_info, nullptr, &instance_);
    if (res != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateInstance failed: %d\n", res);
        return false;
    }
    return true;
}

bool VulkanContext::create_surface_(SDL_Window* window) {
    if (!SDL_Vulkan_CreateSurface(window, instance_, nullptr, &surface_)) {
        std::fprintf(stderr, "[vk] SDL_Vulkan_CreateSurface failed: %s\n",
                     SDL_GetError());
        return false;
    }
    return true;
}

bool VulkanContext::pick_physical_device_() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        std::fprintf(stderr, "[vk] no GPU with Vulkan support\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    // Pick the first device that has a graphics queue and supports presentation.
    for (VkPhysicalDevice dev : devices) {
        uint32_t family = find_graphics_family(dev);
        if (family == UINT32_MAX) continue;

        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, family, surface_, &present_support);
        if (!present_support) continue;

        // Check swapchain extension support.
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
        std::fprintf(stderr, "[vk] GPU: %s\n", props.deviceName);
        return true;
    }

    std::fprintf(stderr, "[vk] no suitable GPU found\n");
    return false;
}

bool VulkanContext::create_device_() {
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = graphics_family_;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    const char* device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pQueueCreateInfos = &queue_ci;
    ci.queueCreateInfoCount = 1;
    ci.ppEnabledExtensionNames = device_exts;
    ci.enabledExtensionCount = 1;
    ci.pEnabledFeatures = &features;

    VkResult res = vkCreateDevice(physical_device_, &ci, nullptr, &device_);
    if (res != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateDevice failed: %d\n", res);
        return false;
    }
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
    return true;
}

bool VulkanContext::create_swapchain_(int width, int height) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);

    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data());

    uint32_t mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, nullptr);
    std::vector<VkPresentModeKHR> modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, modes.data());

    VkSurfaceFormatKHR format = pick_surface_format(formats);
    VkPresentModeKHR present_mode = pick_present_mode(modes);

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
    ci.imageFormat = format.format;
    ci.imageColorSpace = format.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = present_mode;
    ci.clipped = VK_TRUE;

    VkResult res = vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_);
    if (res != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateSwapchainKHR failed: %d\n", res);
        return false;
    }

    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    swapchain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());

    swapchain_format_ = format.format;
    swapchain_extent_ = extent;
    return true;
}

bool VulkanContext::create_image_views_() {
    swapchain_image_views_.resize(swapchain_images_.size());
    for (size_t i = 0; i < swapchain_images_.size(); ++i) {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = swapchain_images_[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = swapchain_format_;
        ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = 1;

        VkResult res = vkCreateImageView(device_, &ci, nullptr, &swapchain_image_views_[i]);
        if (res != VK_SUCCESS) {
            std::fprintf(stderr, "[vk] vkCreateImageView failed: %d\n", res);
            return false;
        }
    }
    return true;
}

bool VulkanContext::create_command_pool_() {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphics_family_;

    VkResult res = vkCreateCommandPool(device_, &ci, nullptr, &command_pool_);
    if (res != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = command_pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(device_, &alloc, &command_buffer_);
    return res == VK_SUCCESS;
}

bool VulkanContext::create_sync_objects_() {
    VkSemaphoreCreateInfo sem_ci{};
    sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(device_, &sem_ci, nullptr, &image_available_) != VK_SUCCESS) return false;

    // One render_finished semaphore per swapchain image — see header comment.
    render_finished_.resize(swapchain_images_.size());
    for (auto& sem : render_finished_) {
        if (vkCreateSemaphore(device_, &sem_ci, nullptr, &sem) != VK_SUCCESS) return false;
    }

    if (vkCreateFence(device_, &fence_ci, nullptr, &in_flight_) != VK_SUCCESS) return false;
    return true;
}

void VulkanContext::present_frame() {
    vkWaitForFences(device_, 1, &in_flight_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &in_flight_);

    uint32_t image_index;
    VkResult res = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                         image_available_, VK_NULL_HANDLE, &image_index);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        return;  // swapchain out of date — would need recreation
    }

    vkResetCommandBuffer(command_buffer_, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer_, &begin);

    // Transition layout and clear — minimal work to keep the window alive.
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchain_images_[image_index];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(command_buffer_,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkClearColorValue clear_color = {{0.05f, 0.05f, 0.08f, 1.0f}};  // dark background
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;
    vkCmdClearColorImage(command_buffer_, swapchain_images_[image_index],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);

    // Transition to present layout.
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(command_buffer_,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(command_buffer_);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore wait_sems[] = {image_available_};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = wait_sems;
    submit.pWaitDstStageMask = wait_stages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    // Use the semaphore belonging to this swapchain image to avoid reuse conflicts
    // with the present operation (which is not fence-synchronized).
    VkSemaphore signal_sem = render_finished_[image_index];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signal_sem;

    vkQueueSubmit(graphics_queue_, 1, &submit, in_flight_);

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &signal_sem;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &image_index;

    vkQueuePresentKHR(graphics_queue_, &present);
}

VulkanContext::~VulkanContext() {
    if (device_) vkDeviceWaitIdle(device_);

    if (in_flight_)       vkDestroyFence(device_, in_flight_, nullptr);
    for (auto sem : render_finished_) {
        if (sem) vkDestroySemaphore(device_, sem, nullptr);
    }
    if (image_available_) vkDestroySemaphore(device_, image_available_, nullptr);

    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);

    for (auto iv : swapchain_image_views_) {
        vkDestroyImageView(device_, iv, nullptr);
    }
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    if (device_)    vkDestroyDevice(device_, nullptr);
    if (surface_)   vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_)  vkDestroyInstance(instance_, nullptr);
}

} // namespace world
} // namespace winefox
