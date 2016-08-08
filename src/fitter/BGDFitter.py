from PIL import Image
from numpy import zeros, nonzero

from .ModelFitter import ModelFitter
from src import Face


class BGDFitter(ModelFitter):
    """Batch gradient descent."""
    def __init__(self, image, dimensions=199, model=None,
                 dx=.1, step=100., max_loops=10, callback=None,
                 initial=None, initial_face=None):

        self.__dx = dx
        self.__step = step
        self.__face = None
        self.__current_step = -5
        self.__callback = callback

        self.__derivatives = None
        self.__left_derivatives = None
        self.__right_derivatives = None

        self.__loop = 0
        self.__max_loops = max_loops

        super(BGDFitter, self).__init__(image, dimensions, model, initial,
                                        initial_face)

    def start(self):
        self.__face = Face(coefficients=self._initial.copy(),
                           directed_light=(0, 0, 0), ambient_light=1.0)

        derivatives_size = self._dimensions + 4
        self.__derivatives = zeros(derivatives_size, dtype='f')
        self.__left_derivatives = zeros(derivatives_size, dtype='f')
        self.__right_derivatives = zeros(derivatives_size, dtype='f')

        self.__next_iteration()

    def receive_image(self, image, index=None):
        if index == 'start_iteration':
            shadows = image
            index = 'start'
            if self.__loop >= self.__max_loops:
                self.__finish(shadows)
                return
        if index == 'start' and self.__current_step >= self._dimensions - 1:
            self.__next_iteration()
            return
        elif index == 'start':
            self.__current_step += 1

        self.__get_derivative(self.__current_step, index, image)

    def __next_iteration(self):
        self.__loop += 1
        self.__current_step = -5
        coefficients = (self.__face.coefficients
                - self.__step * self.__derivatives[:self._dimensions])
        directed_light = (self.__face.directed_light
                - self.__step * self.__derivatives[-3:] / 50)
        ambient_light = (self.__face.ambient_light
                - self.__step * self.__derivatives[-4] / 50)
        self.__face = Face(coefficients=coefficients,
                           directed_light=directed_light,
                           ambient_light=ambient_light)
        # print(self.__derivatives)
        self.request_face(self.__face, 'start_iteration')

    def __get_derivative(self, param, step, image):
        if step == 'start':
            shadows = image
            self.__derivatives[param] = self.get_image_deviation(shadows)

            face = self.__derivative_face(param, self.__dx)

            self.request_face(face, 'right_derivative')
        elif 'derivative' in step:
            shadows = image

            face = self.__face

            if 'right' in step:
                face = self.__derivative_face(param, -self.__dx)

                action = 'left_derivative'
                self.__right_derivatives[param] = self.__derivative(
                    self.__derivatives[param],
                    self.get_image_deviation(shadows))
            elif 'left' in step:
                action = 'start'
                self.__left_derivatives[param] = self.__derivative(
                    self.get_image_deviation(shadows),
                    self.__derivatives[param])
                self.__derivatives[param] = (
                    0.5 * (self.__left_derivatives[param]
                           + self.__right_derivatives[param]))

            self.request_face(face, action)

    def __derivative_face(self, param, dx):
        ambient_light = self.__face.ambient_light

        directed_light = self.__face.directed_light.copy()

        coefficients = self.__face.coefficients.copy()

        if param >= 0:
            coefficients[param] += dx
        elif param == -4:
            ambient_light += dx / 2
        else:
            directed_light[param+3] += dx / 2
        # print(coefficients)

        return Face(coefficients=coefficients,
                    directed_light=directed_light,
                    ambient_light=ambient_light)

    def __derivative(self, y0, y1):
        return (y1 - y0) / self.__dx

    def __finish(self, shadows):
        img = shadows
        img[shadows[:, 3] == 0.] = 1.
        img = img[::-1]
        image = Image.new('L', (500, 500))
        image.putdata((img*255).astype('i'))
        image.save('img.png'.format(self.__loop))
        image.close()
        if self.__callback is not None:
            self.__callback(self.__face)
        # print('Finished')
