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
#include <functional>
#include <memory>
#include <thread>

#include "example_interfaces/srv/add_two_ints.hpp"
#include "rclcpp/rclcpp.hpp"

class MinimalService : public rclcpp::Node
{
public:
  MinimalService()
  : Node("minimal_service")
  {
    using namespace std::placeholders;

    service_ = create_service<example_interfaces::srv::AddTwoInts>(
      "add_two_ints",
      std::bind(&MinimalService::add, this, _1, _2));
  }

private:
  void add(
    const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
    example_interfaces::srv::AddTwoInts::Response::SharedPtr response)
  {
    response->sum = request->a + request->b;
    RCLCPP_INFO(
      get_logger(), "Incoming request\na: %ld, b: %ld\nReturning sum: %ld",
      request->a, request->b, response->sum);
  }
  rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
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