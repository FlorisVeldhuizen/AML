# matplotlib inline
import matplotlib.pyplot as plt
import pylab
import numpy as np
import PIL.Image
import keras

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


def blackwhite(img):
    grayimg = np.array(PIL.Image.fromarray(img).convert('L'))
    # normalizing the 255 grayscale to either [0,1].
    grayimg[grayimg == 87] = 0
    grayimg[grayimg != 0] = 1
    return grayimg


def downsample(img):
    img = img[::2, ::2]
    return img


def combosanitize(img):
    img = pongcrop(img)
    # img = grayscale(img)
    img = downsample(img)
    img = blackwhite(img)
    print(img.shape)
    return img


# Keras Convolutional Network
def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (4, 105, 80)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.convolutional.Convolution2D(
        16, 8, 8, subsample=(4, 4), activation='relu'
    )(frames_input)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.convolutional.Convolution2D(
        32, 4, 4, subsample=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.Multiply()([output, actions_input])

    # // self is not yet implemented, commenting out next bit:
    # self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    # optimizer = optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    # self.model.compile(optimizer, loss='mse')


# Render
env.render()
rangelimit = 26
for i in range(rangelimit):
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render
    env.render()
    frame = combosanitize(frame)
    plt.imshow(frame, cmap='gray')
    if i > rangelimit - 1:
        plt.show()

# env.close()  #Add this to stop it from crashing due to sys meta Path is None

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
plt.show()
