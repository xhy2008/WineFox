// vulkan_app.h — Full Vulkan rendering pipeline for the test sandbox.
//
// Creates a complete rendering pipeline:
//   instance + debug messenger → surface → physical device → logical device
//   → swapchain → render pass (color + depth) → graphics pipeline
//   → framebuffers → vertex/index buffers (cube) → command buffers
//   → sync objects → draw frame
//
// MVP matrix is sent via push constants (64 bytes, no descriptor sets needed).
// Double-buffered (MAX_FRAMES_IN_FLIGHT = 2) for overlap between GPU and CPU.

#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <vector>

#include "camera.h"

namespace vkt {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
};

class VulkanApp {
public:
    bool init(SDL_Window* window, int width, int height);
    ~VulkanApp();

    // Render one frame using the current camera state.
    void draw_frame();

    // Call on window resize to recreate the swapchain.
    void on_resize(int width, int height);

    bool ready() const { return device_ != VK_NULL_HANDLE; }

    Camera& camera() { return camera_; }

private:
    // --- Core objects ---
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphics_queue_ = VK_NULL_HANDLE;
    uint32_t         graphics_family_ = 0;

    // --- Swapchain ---
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkFormat         swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D       swapchain_extent_ = {0, 0};
    std::vector<VkImage>     swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;

    // --- Depth buffer ---
    VkImage        depth_image_       = VK_NULL_HANDLE;
    VkDeviceMemory depth_image_memory_ = VK_NULL_HANDLE;
    VkImageView    depth_image_view_  = VK_NULL_HANDLE;
    VkFormat       depth_format_      = VK_FORMAT_UNDEFINED;

    // --- Render pass + pipeline ---
    VkRenderPass     render_pass_     = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_        = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    // --- Buffers ---
    VkBuffer       vertex_buffer_       = VK_NULL_HANDLE;
    VkDeviceMemory vertex_buffer_memory_ = VK_NULL_HANDLE;
    VkBuffer       index_buffer_        = VK_NULL_HANDLE;
    VkDeviceMemory index_buffer_memory_  = VK_NULL_HANDLE;
    uint32_t       index_count_         = 0;

    // --- Command buffers ---
    VkCommandPool                command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;

    // --- Sync objects (per frame in flight) ---
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<VkSemaphore> image_available_;
    std::vector<VkSemaphore> render_finished_;
    std::vector<VkFence>     in_flight_;
    uint32_t                 current_frame_   = 0;
    uint32_t                 image_index_     = 0;
    bool                     framebuffer_resized_ = false;

    // --- Camera ---
    Camera camera_;

    // --- Init helpers ---
    bool create_instance_();
    bool setup_debug_messenger_();
    bool create_surface_(SDL_Window* window);
    bool pick_physical_device_();
    bool create_device_();
    bool create_swapchain_(int width, int height);
    bool create_image_views_();
    bool find_depth_format_();
    bool create_depth_resources_();
    bool create_render_pass_();
    bool create_pipeline_();
    bool create_framebuffers_();
    bool create_command_pool_();
    bool create_vertex_buffer_();
    bool create_index_buffer_();
    bool create_command_buffers_();
    bool create_sync_objects_();

    // --- Helpers ---
    uint32_t find_memory_type_(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    bool find_supported_format_(const std::vector<VkFormat>& candidates,
                                VkImageTiling tiling,
                                VkFormatFeatureFlags features,
                                VkFormat& out) const;
    void record_command_buffer_(VkCommandBuffer cmd, uint32_t image_index, const glm::mat4& mvp);
    void recreate_swapchain_(int width, int height);
    void cleanup_swapchain_();
    void create_buffer_(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags props,
                        VkBuffer& buffer, VkDeviceMemory& memory) const;
    void copy_buffer_(VkBuffer src, VkBuffer dst, VkDeviceSize size) const;
    VkShaderModule load_shader_(const char* path) const;
};

} // namespace vkt
