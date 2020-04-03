#include "catch.hpp"
#include "reward.h"
#include "model.h"

TEST_CASE("calculate_rewards") {
    Model m(0.0);
    StateType s;
    ActionType a;
    m.record_action(s, a, 0.0, 0.0); 
    m.record_action(s, a, 1.0, 0.0); 
    m.record_action(s, a, 2.0, 1.0); 

    calculate_rewards(m, 0.0);

    REQUIRE(m.rewards[0] == 0.0);
    REQUIRE(m.rewards[1] == 1.0);
    REQUIRE(m.rewards[2] == 2.0);

    REQUIRE(m.normalized_rewards[0] == Approx(0.0));
    REQUIRE(m.normalized_rewards[1] == Approx(1.0 / sqrt(2.0/3.0)));
    REQUIRE(m.normalized_rewards[2] == Approx(2.0 / sqrt(2.0/3.0)));

	REQUIRE(m.adv[0] == Approx((2 - 3*sqrt(6))/(2.*sqrt(11 - 3*sqrt(6)))));
	REQUIRE(m.adv[1] == Approx(1/sqrt(11 - 3*sqrt(6))));
	REQUIRE(m.adv[2] == Approx((-4 + 3*sqrt(6))/sqrt(44 - 12*sqrt(6))));

}