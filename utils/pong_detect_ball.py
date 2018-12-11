import pdb
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_ball_pos(og_image):
    top_crop = 35
    bottom_crop = 194
    image = og_image[top_crop:bottom_crop]  # crop

    boundaries = [[235, 235, 235], [237, 237, 237]]
    lower, upper = boundaries

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    mask = cv2.inRange(image, lower, upper)

    # get the ball center
    return np.nonzero(mask), image.shape


def ball_half_screen_position(og_image):
    """Return 1 if the ball is in the top half of the screen, 0 if the ball
    is in the bottom half of the screen, and -1 if the ball is not visible."""
    mask, image_shape = get_ball_pos(og_image)
    ball_y, ball_x = mask
    img_y, img_x, img_d = image_shape

    if len(ball_y) == 0:
        return -1
    else:
        ball_y_center = np.mean(ball_y)
        return ball_y_center < img_y / 2.


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    env.reset()
    for i in range(1000):
        env.step(0)

        if i % 20 == 0:
            og_image = env.unwrapped._get_obs()


            pos = ball_half_screen_position(og_image)
            if pos == 1:
                print('upper')
            elif pos == 0:
                print('lower')
            else:
                print('no ball')

            plt.imshow(og_image)
            plt.show()

            # print('uncropped')
            # plt.imshow(image)
            # plt.show()
            # top_crop = 35
            # bottom_crop = 194
            # image = og_image[top_crop:bottom_crop]  # crop
            # img_y, img_x, img_d = image.shape
            # # print('cropped')

            # boundaries = [[235, 235, 235], [237, 237, 237]]
            # lower, upper = boundaries

            # # create NumPy arrays from the boundaries
            # lower = np.array(lower, dtype = "uint8")
            # upper = np.array(upper, dtype = "uint8")

            # # find the colors within the specified boundaries and apply
            # # the mask
            # mask = cv2.inRange(image, lower, upper)
            # # output = cv2.bitwise_and(image, image, mask = mask)

            # # get the ball center
            # y = np.nonzero(mask)[0]
            # y_center = np.mean(y)
            # print('y_center:', y_center)
            
            # # classify
            # if len(y) != 0:
            #     if y_center > img_y / 2.:
            #         print('lower')
            #     else:
            #         print('higher')
            #     plt.imshow(og_image)
            #     plt.show()
            #     pdb.set_trace()
            # else:
            #     print('No ball')


            print()
