// camera.h — First-person camera for room exploration.
//
// FPS-style camera for navigating inside the enclosed room:
//   Left-drag   : look around (yaw + pitch)
//   W/A/S/D     : move forward / left / back / right
//   Q / E       : move down / up
//   Mouse wheel : adjust movement speed

#pragma once

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vkt {

class Camera {
public:
    Camera() = default;

    void set_perspective(float fov_deg, float aspect, float near_plane, float far_plane) {
        fov_    = glm::radians(fov_deg);
        aspect_ = aspect;
        near_   = near_plane;
        far_    = far_plane;
    }

    void set_aspect(float aspect) { aspect_ = aspect; }
    void set_position(const glm::vec3& pos) { position_ = pos; }

    // Look-around via mouse drag.
    //   dx > 0 (mouse right) → look right (yaw increases)
    //   dy > 0 (mouse down)  → look down  (pitch decreases)
    void rotate(float dx, float dy) {
        yaw_   += dx * sensitivity_;
        pitch_ -= dy * sensitivity_;
        pitch_  = glm::clamp(pitch_, -1.55f, 1.55f);
    }

    // Movement along local axes. delta is in seconds (frame time).
    // Uses flat forward (XZ plane) so looking up/down doesn't affect walking.
    void move_forward(float delta) { position_ += forward_flat() * delta * speed_; }
    void move_right(float delta)   { position_ += right() * delta * speed_; }
    void move_up(float delta)      { position_ += world_up_ * delta * speed_; }

    void adjust_speed(float delta) {
        speed_ = glm::clamp(speed_ + delta * 0.5f, 0.5f, 20.0f);
    }

    glm::vec3 position() const { return position_; }
    float     speed() const    { return speed_; }

    glm::mat4 view() const {
        return glm::lookAt(position_, position_ + forward(), world_up_);
    }

    glm::mat4 projection() const {
        // Vulkan NDC: Y axis points DOWN (framebuffer origin is top-left).
        // glm::perspective produces OpenGL-style (Y up), so flip Y.
        glm::mat4 p = glm::perspective(fov_, aspect_, near_, far_);
        p[1][1] *= -1.0f;
        return p;
    }

    glm::mat4 view_projection() const {
        return projection() * view();
    }

private:
    glm::vec3 forward() const {
        float cp = cos(pitch_), sp = sin(pitch_);
        float sy = sin(yaw_),   cy = cos(yaw_);
        return glm::normalize(glm::vec3(cp * sy, sp, -cp * cy));
    }

    glm::vec3 forward_flat() const {
        float sy = sin(yaw_), cy = cos(yaw_);
        return glm::normalize(glm::vec3(sy, 0.0f, -cy));
    }

    glm::vec3 right() const {
        return glm::normalize(glm::cross(forward_flat(), world_up_));
    }

    // Perspective
    float fov_    = glm::radians(45.0f);
    float aspect_ = 1.0f;
    float near_   = 0.1f;
    float far_    = 100.0f;

    // Position & orientation
    glm::vec3 position_ = glm::vec3(0.0f, 0.0f, 0.0f);
    float yaw_   = 0.0f;  // left-right
    float pitch_ = 0.0f;  // up-down

    // Controls
    float sensitivity_ = 0.005f;
    float speed_       = 3.0f;  // units per second

    glm::vec3 world_up_ = glm::vec3(0.0f, 1.0f, 0.0f);
};

} // namespace vkt
