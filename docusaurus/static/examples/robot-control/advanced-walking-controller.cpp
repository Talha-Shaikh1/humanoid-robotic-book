/*
 * Advanced Humanoid Walking Gait Controller (C++)
 *
 * This file implements an advanced walking gait controller for humanoid robots,
 * incorporating ZMP (Zero Moment Point) control, inverse kinematics, and balance
 * maintenance for stable locomotion in C++ for ROS2.
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>
#include <string>

using namespace std::chrono_literals;

class ZMPPreviewController {
private:
    double dt_;  // Control timestep
    int preview_steps_;  // Number of steps for preview control

    // Robot parameters
    double com_height_;
    double gravity_;

    // Control parameters
    double omega_;
    Eigen::Matrix3d A_;
    Eigen::Vector3d B_;
    Eigen::RowVector3d K_;  // Feedback gain matrix

    // State vector [x, dx, zmp_error_integral]
    Eigen::Vector3d state_;

public:
    ZMPPreviewController(double dt = 0.01, double preview_window = 2.0)
        : dt_(dt), com_height_(0.65), gravity_(9.81) {

        preview_steps_ = static_cast<int>(preview_window / dt_);

        // Calculate omega for ZMP control
        omega_ = std::sqrt(gravity_ / com_height_);

        // Initialize system matrices
        A_ << 1, dt_, (std::cosh(omega_*dt_) - 1) / (omega_*omega_),
              0, 1, std::sinh(omega_*dt_) / omega_,
              omega_*omega_, 0, std::cosh(omega_*dt_);

        B_ << (omega_*dt_ - std::sinh(omega_*dt_)) / (omega_*omega_*omega_),
              (1 - std::cosh(omega_*dt_)) / (omega_*omega_),
              std::sinh(omega_*dt_) / omega_;

        // Initialize feedback gain (simplified for demonstration)
        K_ << -2.0, -2.0, -0.5;

        // Initialize state
        state_ << 0.0, 0.0, 0.0;
    }

    /**
     * Update the ZMP controller with new measurements and references.
     *
     * @param desired_zmp Desired ZMP position [x, y]
     * @param measured_zmp Measured ZMP position [x, y]
     * @param reference_com_x Reference center of mass x position
     * @return Next COM state for trajectory generation
     */
    std::pair<double, double> update(
        const std::pair<double, double>& desired_zmp,
        const std::pair<double, double>& measured_zmp,
        double reference_com_x
    ) {
        // Calculate ZMP error
        double zmp_error_x = desired_zmp.first - measured_zmp.first;

        // Update integral term in state
        state_[2] += zmp_error_x * dt_;

        // Calculate control input using state feedback
        double control_input = -K_ * state_;

        // Update state using system dynamics
        state_ = A_ * state_ + B_ * control_input;

        // Return next COM position based on state
        double next_com_x = state_[0];
        double next_com_y = reference_com_x;  // Simplified - maintain y reference

        return std::make_pair(next_com_x, next_com_y);
    }

    /**
     * Reset the controller state.
     */
    void reset() {
        state_.setZero();
    }
};

class InverseKinematicsSolver {
private:
    double upper_leg_length_;
    double lower_leg_length_;
    double foot_length_;

public:
    InverseKinematicsSolver(double upper_leg = 0.35, double lower_leg = 0.35, double foot = 0.15)
        : upper_leg_length_(upper_leg), lower_leg_length_(lower_leg), foot_length_(foot) {}

    /**
     * Solve inverse kinematics for a single leg.
     *
     * @param target_pos Target foot position [x, y, z] relative to hip
     * @param leg_side 'left' or 'right' for coordinate system adjustment
     * @return Joint angles [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
     */
    std::vector<double> solveLegIK(
        const std::vector<double>& target_pos,
        const std::string& leg_side = "left"
    ) {
        double x = target_pos[0];
        double y = target_pos[1];
        double z = target_pos[2];

        // Calculate hip yaw angle
        double hip_yaw = (std::sqrt(x*x + y*y) > 0.01) ? std::atan2(y, x) : 0.0;

        // Project to 2D plane for remaining calculations
        double r_horiz = std::sqrt(x*x + y*y);  // Horizontal distance from hip to foot
        double r_total = std::sqrt(r_horiz*r_horiz + z*z);  // 3D distance from hip to foot

        // Hip roll compensation for lateral movement
        double hip_roll = std::atan2(-z, r_horiz);  // Simplified calculation

        // Calculate knee angle using law of cosines
        double cos_knee = (upper_leg_length_*upper_leg_length_ +
                          lower_leg_length_*lower_leg_length_ -
                          r_total*r_total) /
                         (2 * upper_leg_length_ * lower_leg_length_);

        // Clamp to valid range for acos
        cos_knee = std::max(-1.0, std::min(1.0, cos_knee));
        double knee_angle = M_PI - std::acos(cos_knee);

        // Calculate hip pitch angle
        // First, calculate angle between upper leg and line from hip to foot
        double alpha = std::acos((upper_leg_length_*upper_leg_length_ + r_total*r_total -
                                 lower_leg_length_*lower_leg_length_) /
                                (2 * upper_leg_length_ * r_total));

        // Hip pitch is the angle from vertical minus alpha
        double hip_pitch = std::atan2(z, r_horiz) + alpha;

        // Calculate ankle angles to maintain foot orientation
        // Simplified: ankle compensates for knee and hip angles
        double ankle_pitch = -(hip_pitch + (M_PI - knee_angle));
        double ankle_roll = -hip_roll;  // Compensate for hip roll

        // Adjust for leg side (left/right symmetry)
        double multiplier = (leg_side == "left") ? 1.0 : -1.0;

        return {
            hip_yaw,                    // Hip yaw
            hip_roll * multiplier,      // Hip roll (inverted for right leg)
            hip_pitch,                  // Hip pitch
            knee_angle,                 // Knee angle
            ankle_pitch,                // Ankle pitch
            ankle_roll * multiplier     // Ankle roll (inverted for right leg)
        };
    }
};

class WalkingPatternGenerator {
private:
    double step_length_;
    double step_height_;
    double step_duration_;
    double dt_;

    // Walking parameters
    double stride_length_;
    double step_width_;
    double swing_phase_ratio_;

public:
    WalkingPatternGenerator(double step_length = 0.3, double step_height = 0.05,
                           double step_duration = 0.8)
        : step_length_(step_length), step_height_(step_height),
          step_duration_(step_duration), dt_(0.01), step_width_(0.18),
          swing_phase_ratio_(0.3) {}

    /**
     * Generate smooth trajectory for a single step.
     *
     * @param start_pos Starting foot position [x, y, z]
     * @param target_pos Target foot position [x, y, z]
     * @param current_time Current time in the step cycle
     * @param total_time Total time for this step
     * @return Current foot position along the trajectory
     */
    std::vector<double> generateStepTrajectory(
        const std::vector<double>& start_pos,
        const std::vector<double>& target_pos,
        double current_time,
        double total_time
    ) {
        // Calculate phase (0.0 to 1.0) of the step
        double phase = std::min(1.0, current_time / total_time);

        // Calculate stance/swing phases
        double swing_start = (1.0 - swing_phase_ratio_) / 2.0;
        double swing_end = swing_start + swing_phase_ratio_;

        double pos_x, pos_y, pos_z;

        if (phase < swing_start) {
            // Early stance phase - interpolate to midpoint
            double t_interp = phase / swing_start;
            // Use quadratic interpolation for smooth start
            pos_x = start_pos[0] + (target_pos[0] - start_pos[0]) * 0.5 * t_interp * t_interp;
            pos_y = start_pos[1] + (target_pos[1] - start_pos[1]) * 0.5 * t_interp * t_interp;
            pos_z = start_pos[2];  // Keep foot on ground
        } else if (phase <= swing_end) {
            // Swing phase - move to target with lift
            double t_swing = (phase - swing_start) / swing_phase_ratio_;

            // Horizontal movement using sinusoidal interpolation
            pos_x = start_pos[0] + (target_pos[0] - start_pos[0]) *
                   (0.5 + 0.5 * std::sin(M_PI * (t_swing - 0.5)));
            pos_y = start_pos[1] + (target_pos[1] - start_pos[1]) *
                   (0.5 + 0.5 * std::sin(M_PI * (t_swing - 0.5)));

            // Vertical lift - sinusoidal profile
            if (t_swing < 0.5) {
                pos_z = start_pos[2] + step_height_ * std::sin(M_PI * t_swing);
            } else {
                pos_z = target_pos[2] + step_height_ * std::sin(M_PI * (1.0 - t_swing));
            }
        } else {
            // Late stance phase - hold at target position with smooth landing
            double t_interp = (phase - swing_end) / (1.0 - swing_end);
            pos_x = target_pos[0];
            pos_y = target_pos[1];
            pos_z = target_pos[2] + (start_pos[2] - target_pos[2]) * t_interp;  // Smooth landing
        }

        return {pos_x, pos_y, pos_z};
    }

    /**
     * Generate Center of Mass trajectory for stable walking.
     *
     * @param walk_params Walking parameters
     * @return Smooth COM trajectory
     */
    std::vector<std::vector<double>> generateCOMTrajectory(
        const std::map<std::string, double>& walk_params
    ) {
        // Simplified: generate a smooth body trajectory that moves forward
        // while maintaining balance over the supporting foot
        std::vector<std::vector<double>> com_trajectory;

        // For now, return an empty trajectory - would be implemented with actual walking pattern
        return com_trajectory;
    }

    // Getters for walking parameters
    double getStepWidth() const { return step_width_; }
    double getStrideLength() const { return stride_length_; }
};

class AdvancedWalkingController : public rclcpp::Node {
private:
    // Controller components
    std::unique_ptr<ZMPPreviewController> zmp_controller_;
    std::unique_ptr<InverseKinematicsSolver> ik_solver_;
    std::unique_ptr<WalkingPatternGenerator> pattern_generator_;

    // Walking state
    bool is_walking_;
    std::vector<double> walk_velocity_;  // [x, y, theta velocities]
    double current_step_time_;
    double total_step_time_;
    std::string support_leg_;  // "left" or "right"
    int step_count_;

    // Robot state
    std::map<std::string, double> joint_states_;
    sensor_msgs::msg::Imu::SharedPtr imu_data_;
    std::vector<double> com_estimate_;  // Estimated center of mass [x, y, z]

    // Publishers and subscribers
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_cmd_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    // Control timer
    rclcpp::TimerBase::SharedPtr control_timer_;

public:
    AdvancedWalkingController() : Node("advanced_walking_controller") {
        // Initialize controller components
        zmp_controller_ = std::make_unique<ZMPPreviewController>(0.01, 2.0);
        ik_solver_ = std::make_unique<InverseKinematicsSolver>(0.35, 0.35, 0.15);
        pattern_generator_ = std::make_unique<WalkingPatternGenerator>(0.3, 0.05, 0.8);

        // Initialize walking state
        is_walking_ = false;
        walk_velocity_ = {0.0, 0.0, 0.0};
        current_step_time_ = 0.0;
        total_step_time_ = 0.8;
        support_leg_ = "left";
        step_count_ = 0;

        // Initialize robot state
        com_estimate_ = {0.0, 0.0, 0.65};

        // Create publishers and subscribers
        joint_cmd_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_commands", 10
        );

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&AdvancedWalkingController::jointStateCallback, this, std::placeholders::_1)
        );

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 10,
            std::bind(&AdvancedWalkingController::imuCallback, this, std::placeholders::_1)
        );

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&AdvancedWalkingController::cmdVelCallback, this, std::placeholders::_1)
        );

        // Create control timer (100Hz)
        control_timer_ = this->create_wall_timer(
            10ms, std::bind(&AdvancedWalkingController::controlLoop, this)
        );

        RCLCPP_INFO(this->get_logger(), "Advanced Walking Controller initialized");
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        // Update joint states
        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (i < msg->position.size()) {
                joint_states_[msg->name[i]] = msg->position[i];
            }
        }
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // Update IMU data
        imu_data_ = msg;
    }

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Update walking velocity commands
        walk_velocity_ = {msg->linear.x, msg->linear.y, msg->angular.z};
    }

    std::pair<double, double> estimateZMP() {
        /*
         * Estimate Zero Moment Point from current robot state.
         * In a real implementation, this would use force/torque sensors.
         * For simulation, we'll estimate from IMU data.
         */
        if (imu_data_ != nullptr) {
            // Extract roll and pitch from IMU quaternion
            tf2::Quaternion quat(
                imu_data_->orientation.x,
                imu_data_->orientation.y,
                imu_data_->orientation.z,
                imu_data_->orientation.w
            );

            double roll, pitch, yaw;
            tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

            // Simplified ZMP estimation based on tilt
            double zmp_x = -com_estimate_[2] * std::tan(pitch);
            double zmp_y = com_estimate_[2] * std::tan(roll);

            return std::make_pair(zmp_x, zmp_y);
        }

        // Default if no IMU data
        return std::make_pair(0.0, 0.0);
    }

    std::vector<std::vector<double>> calculateSupportPolygon() {
        /*
         * Calculate the support polygon based on foot positions.
         * Simplified implementation assuming rectangular feet.
         */
        std::vector<std::vector<double>> support_polygon;

        if (support_leg_ == "left") {
            // Right foot is in air, left foot provides support
            support_polygon = {
                {-0.1, -0.09},  // Left foot corners
                {0.1, -0.09},
                {0.1, 0.09},
                {-0.1, 0.09}
            };
        } else {
            // Left foot is in air, right foot provides support
            double step_width = pattern_generator_->getStepWidth();
            support_polygon = {
                {-0.1, -0.09 - step_width},  // Right foot corners
                {0.1, -0.09 - step_width},
                {0.1, 0.09 - step_width},
                {-0.1, 0.09 - step_width}
            };
        }

        return support_polygon;
    }

    bool isZMPSafe(const std::pair<double, double>& zmp_pos,
                   const std::vector<std::vector<double>>& support_polygon) {
        /*
         * Check if the ZMP is within the support polygon.
         * Simplified point-in-polygon check for rectangular support area.
         */
        double zmp_x = zmp_pos.first;
        double zmp_y = zmp_pos.second;

        // Get bounding box of support polygon
        double min_x = support_polygon[0][0];
        double max_x = support_polygon[0][0];
        double min_y = support_polygon[0][1];
        double max_y = support_polygon[0][1];

        for (const auto& point : support_polygon) {
            min_x = std::min(min_x, point[0]);
            max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]);
            max_y = std::max(max_y, point[1]);
        }

        return (min_x <= zmp_x && zmp_x <= max_x && min_y <= zmp_y && zmp_y <= max_y);
    }

    void controlLoop() {
        if (!is_walking_) {
            return;
        }

        // Update step timing
        current_step_time_ += 0.01;
        if (current_step_time_ >= total_step_time_) {
            // Complete current step, switch support leg
            current_step_time_ = 0.0;
            support_leg_ = (support_leg_ == "left") ? "right" : "left";
            step_count_++;
        }

        // Estimate current ZMP
        auto current_zmp = estimateZMP();

        // Calculate desired ZMP based on walking pattern
        double step_phase = current_step_time_ / total_step_time_;
        double desired_zmp_x = walk_velocity_[0] * 0.2;  // Proportional to velocity
        double desired_zmp_y = 0.0;  // Try to maintain centered ZMP

        // Adjust for turning
        if (std::abs(walk_velocity_[2]) > 0.01) {
            double turn_effect = walk_velocity_[2] * 0.1;  // Simplified turning effect
            desired_zmp_y = turn_effect;
        }

        auto desired_zmp = std::make_pair(desired_zmp_x, desired_zmp_y);

        // Get support polygon
        auto support_poly = calculateSupportPolygon();

        // Check stability
        bool is_stable = isZMPSafe(current_zmp, support_poly);

        // Update ZMP controller
        auto reference_com = std::make_pair(0.0, 0.0);  // Reference center of mass position
        auto next_com_state = zmp_controller_->update(desired_zmp, current_zmp, reference_com.first);

        // Generate foot trajectories based on walking pattern
        std::vector<double> left_foot_pos, right_foot_pos;

        if (support_leg_ == "left") {
            // Right foot is swinging
            right_foot_pos = {
                step_count_ * 0.3 + 0.2,  // Forward position
                -pattern_generator_->getStepWidth() / 2.0,  // Lateral position
                0.0  // Height (on ground)
            };

            // Generate swing trajectory for right foot
            right_foot_pos = pattern_generator_->generateStepTrajectory(
                {step_count_ * 0.3, -pattern_generator_->getStepWidth() / 2.0, 0.0},  // Previous position
                right_foot_pos,
                current_step_time_,
                total_step_time_
            );

            // Left foot stays in place
            left_foot_pos = {
                (step_count_ - 1) * 0.3,
                pattern_generator_->getStepWidth() / 2.0,
                0.0
            };
        } else {
            // Left foot is swinging
            left_foot_pos = {
                step_count_ * 0.3 + 0.2,  // Forward position
                pattern_generator_->getStepWidth() / 2.0,  // Lateral position
                0.0  // Height (on ground)
            };

            // Generate swing trajectory for left foot
            left_foot_pos = pattern_generator_->generateStepTrajectory(
                {(step_count_ - 1) * 0.3, pattern_generator_->getStepWidth() / 2.0, 0.0},  // Previous position
                left_foot_pos,
                current_step_time_,
                total_step_time_
            );

            // Right foot stays in place
            right_foot_pos = {
                (step_count_ - 1) * 0.3,
                -pattern_generator_->getStepWidth() / 2.0,
                0.0
            };
        }

        // Solve inverse kinematics for both legs
        auto left_leg_joints = ik_solver_->solveLegIK(left_foot_pos, "left");
        auto right_leg_joints = ik_solver_->solveLegIK(right_foot_pos, "right");

        // Combine with arm movements for balance (simplified)
        std::vector<double> arm_compensation = {0.0, 0.0};  // Simplified arm positions for balance

        // Prepare joint commands
        std::vector<std::string> joint_names = {
            // Left leg
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
            // Right leg
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle_pitch", "right_ankle_roll",
            // Left arm for balance
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
            "left_elbow", "left_wrist_pitch", "left_wrist_yaw",
            // Right arm for balance
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
            "right_elbow", "right_wrist_pitch", "right_wrist_yaw"
        };

        std::vector<double> positions;
        positions.insert(positions.end(), left_leg_joints.begin(), left_leg_joints.end());
        positions.insert(positions.end(), right_leg_joints.begin(), right_leg_joints.end());

        // Add simplified arm positions for balance
        std::vector<double> arm_positions = {-0.2, 0.1, 0.0, -0.5, 0.0, 0.0,  // Left arm
                                              0.2, -0.1, 0.0, -0.5, 0.0, 0.0};  // Right arm
        positions.insert(positions.end(), arm_positions.begin(), arm_positions.end());

        // Create and publish joint trajectory message
        auto joint_trajectory = trajectory_msgs::msg::JointTrajectory();
        joint_trajectory.joint_names = joint_names;

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = positions;
        point.velocities.resize(positions.size(), 0.0);  // Zero velocities for simplicity
        point.accelerations.resize(positions.size(), 0.0);  // Zero accelerations for simplicity

        // Set time from start
        point.time_from_start.sec = 0;
        point.time_from_start.nanosec = 10000000;  // 10ms

        joint_trajectory.points.push_back(point);

        // Publish the joint commands
        joint_cmd_pub_->publish(joint_trajectory);

        // Log stability information
        std::string stability_msg = is_stable ? "STABLE" : "UNSTABLE";
        RCLCPP_DEBUG(
            this->get_logger(),
            "Walk control - Step: %d, Phase: %.2f, "
            "ZMP: (%.3f, %.3f), Desired: (%.3f, %.3f), Stability: %s",
            step_count_, step_phase, current_zmp.first, current_zmp.second,
            desired_zmp.first, desired_zmp.second, stability_msg.c_str()
        );
    }
};

int main(int argc, char * argv[]) {
    /*
     * Main function to run the advanced walking controller.
     */
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AdvancedWalkingController>());
    rclcpp::shutdown();
    return 0;
}