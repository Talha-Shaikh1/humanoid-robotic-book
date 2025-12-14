// Copyright 2023 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class MinimalService : public rclcpp::Node
{
public:
  MinimalService()
  : Node("minimal_service")
  {
    using namespace std::placeholders;

    server_ = this->create_service<example_interfaces::srv::AddTwoInts>(
      "add_two_ints",
      std::bind(&MinimalService::add, this, _1, _2));
  }

private:
  void add(
    const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
    std::shared_ptr<example_interfaces::srv::AddTwoInts::Response> response)
  {
    response->sum = request->a + request->b;
    RCLCPP_INFO(
      this->get_logger(), "Incoming request\na: %ld b: %ld",
      request->a, request->b);
    RCLCPP_INFO(this->get_logger(), "Sending back response: [%ld]", response->sum);
  }
  rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr server_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<MinimalService>();

  RCLCPP_INFO(node->get_logger(), "Starting minimal service, expecting requests for adding two ints");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}