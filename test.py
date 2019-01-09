# matplotlib inline
import matplotlib.pyplot as plt
import pylab
import numpy as np
import PIL.Image

# Import the gym module
import gym

# Create a breakout environment
env = gym.make("Pong-v0")
# Reset it, returns the starting frame
frame = env.reset()
print("the frame variable is of type ", type(frame))
print("the shape of frame are ", frame.shape)


# Img Sanitizing
def pongcrop(img):
    img = img[34:194, 0:159, :]
    return img


def grayscale(img):
    grayimg = np.array(PIL.Image.fromarray(img).convert('L'))
    print(grayimg.shape)
    return grayimg


def downsample(img):
    img = img[::2, ::2]
    return img


def combosanitize(img):
    img = pongcrop(img)
    img = grayscale(img)
    img = downsample(img)
    return img


# Render
env.render()
rangelimit = 26
for i in range(rangelimit):
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render
    env.render()
    frame = combosanitize(frame)
    plt.imshow(frame)
    if i > rangelimit - 1:
        plt.show()

# env.close()  #Add this to stop it from crashing due to sys meta Path is None

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
plt.show()
