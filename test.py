#matplotlib inline
import matplotlib.pyplot as plt
import pylab
import operator


# Import the gym module
import gym

# Create a breakout environment
env = gym.make("Pong-v0")
# Reset it, returns the starting frame
frame = env.reset()
print("the frame variable is of type ", type(frame))
print("the shape of frame are ", frame.shape)

# Render
env.render()
rangelimit = 21
for i in range(rangelimit):
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render
    env.render()
    plt.imshow(frame)
    if i > rangelimit-1:
        plt.show()

# env.close()  #Add this to stop it from crashing due to sys meta Path is None

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
plt.show()