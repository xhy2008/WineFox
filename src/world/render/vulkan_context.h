// vulkan_context.h — Vulkan device + swapchain initialization.
//
// Creates a VkInstance, VkSurfaceKHR (from SDL window), VkDevice, and
// VkSwapchainKHR. No rendering is performed — this is the scaffolding for
// future Phase C (PBR room rendering). The present loop just acquires
// and presents images to keep the window responsive.
//
// Validation layers are enabled in Debug builds only.

#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <vector>

namespace winefox {
namespace world {

class VulkanContext {
public:
    bool init(SDL_Window* window, int width, int height);
    ~VulkanContext();

    // Present one frame (clear screen to a solid color). This keeps the
    // window responsive without rendering any content.
    void present_frame();

    bool ready() const { return device_ != VK_NULL_HANDLE; }

private:
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphics_queue_ = VK_NULL_HANDLE;
    uint32_t         graphics_family_ = 0;

    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkFormat         swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D       swapchain_extent_ = {0, 0};
    std::vector<VkImage>       swapchain_images_;
    std::vector<VkImageView>   swapchain_image_views_;

    VkCommandPool    command_pool_   = VK_NULL_HANDLE;
    VkCommandBuffer  command_buffer_ = VK_NULL_HANDLE;

    // Synchronization
    // image_available_: signaled by vkAcquireNextImageKHR, waited on by vkQueueSubmit.
    //   Safe to reuse — the in_flight_ fence ensures the previous submit finished
    //   consuming it before we acquire again.
    // render_finished_: one per swapchain image. Signaled by vkQueueSubmit, waited on
    //   by vkQueuePresentKHR. Must NOT be reused across frames because the present
    //   operation is not synchronized by the fence and may still hold the semaphore.
    VkSemaphore      image_available_ = VK_NULL_HANDLE;
    std::vector<VkSemaphore> render_finished_;
    VkFence          in_flight_       = VK_NULL_HANDLE;

    bool create_instance_();
    bool create_surface_(SDL_Window* window);
    bool pick_physical_device_();
    bool create_device_();
    bool create_swapchain_(int width, int height);
    bool create_image_views_();
    bool create_command_pool_();
    bool create_sync_objects_();
};

} // namespace world
} // namespace winefox
