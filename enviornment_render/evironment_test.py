import gfootball.env  as env


""" Scenario names-

1)  11_vs_11_easy_stochastic-
2)  11_vs_11_hard_stochastic
3)  11_vs_11_stochastic
4)  academy_3_vs_1_with_keeper
5)  academy_corner
6)  academy_counterattack_easy
7)  academy_counterattack_hard
8)  academy_empty_goal_close
9)  academy_empty_goal
10) academy_pass_and_shoot_with_keeper
11) academy_run_pass_and_shoot_with_keeper
12) academy_run_to_score
13) academy_run_to_score_with_keeper
14) academy_single_goal_versus_lazy

"""

# the observation is a vector of size 115 
env1 = env.create_environment(env_name ='academy_empty_goal_close',render=True,representation='simple115') 

# the observation is the rendered view of the football field downsampled to w=96, h=72. The observation size is: 72x96x3 (or 72x96x12 when "stacked" is True)
env2 = env.create_environment(env_name ='academy_empty_goal_close',render=True,representation='pixels') 

#the observation is the rendered view of the football field in gray scale and downsampled to w=96, h=72. The observation size is 72x96x1 (or 72x96x4 when stacked is True).
env3 = env.create_environment(env_name ='academy_empty_goal_close',render=True,representation='pixels_gray')

#also referred to as super minimap. The observation is composed of 4 planes of size w=96, h=72.Its size is then 72x96x4 (or 72x96x16 when stacked is True).
env4 = env.create_environment(env_name ='academy_empty_goal_close',render=True,representation='extracted') #for a vector state representation


print(env1.observation_space)
print(env1.action_space)
done = False

while True:
    env1.reset()

    done = False
    while not done:
        action = env1.action_space.sample()
        observation,reward,done,info = env1.step(action)
        #print(action)
        #print(observation)
        #print(reward)
        #print(info)
