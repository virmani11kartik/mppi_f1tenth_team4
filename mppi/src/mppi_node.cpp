#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

class MPPI : public rclcpp::Node
{
    // Implement MPPI
    // This is just a template, you are free to implement your own node!

private:

public:
    MPPI() : Node("MPPI_node")
    {
        // TODO: create ROS subscribers and publishers
    }

    void pose_callback(const geometry_msgs::msg::PoseStamped &pose_msg)
    {
    (void)pose_msg;
    }

    ~MPPI() {}
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPPI>());
    rclcpp::shutdown();
    return 0;
}
