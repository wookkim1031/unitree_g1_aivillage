#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <iostream>
#include <unistd.h>

using namespace unitree::robot;
using LowState = unitree_hg::msg::dds_::LowState_;

void Handler(const void* msg)
{
    auto* s = (const LowState*)msg;
    std::cout << "tick=" << s->tick() << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <interface>" << std::endl;
        return 1;
    }

    ChannelFactory::Instance()->Init(0, argv[1]);

    ChannelSubscriberPtr<LowState> sub(
        new ChannelSubscriber<LowState>("rt/lowstate"));
    sub->InitChannel(Handler, 1);

    while (true) sleep(10);
}