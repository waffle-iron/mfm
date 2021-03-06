from numpy import ones, zeros, linspace, argmin, exp, array
from numpy.random import rand

from src import Face
from .ModelFitter import ModelFitter


class GibbsSamplerFitter(ModelFitter):
    def __init__(self, image, dimensions=199, model=None, steps=None,
                 max_loops=1, determined_loops=0,
                 initial_face=None, callback=None):
        super(GibbsSamplerFitter, self).__init__(image, dimensions, model,
                                                 initial_face, callback)

        self.__steps = ones(dimensions, dtype='i') if steps is None else steps
        self.__current_step = None
        self.__parameters = zeros(dimensions, dtype='f')

        self.__errors = None
        self.__values = None
        self.__loop = 0
        self.__max_loops = max_loops
        self.__determined_loops = determined_loops
        self.__face = None

    def start(self):
        self.__loop = 0

        self.__face = self._initial_face
        self.__parameters = self.__face.as_array

        self.request_face(self.__face, 'init')
        self.__get_parameter(0)

    def __get_parameter(self, i):
        self.__current_step = i

        if self.__steps[i] % 2 == 0:
            self.__values = linspace(-4, 4, self.__steps[i] + 1, dtype='f')
        else:
            self.__values = linspace(-4, 4, self.__steps[i] + 2, dtype='f')

        self.__errors = [None] * self.__values.size
        for i, value in enumerate(self.__values):
            self.__values[i] = value
            parameters = self.__parameters.copy()
            parameters[self.__current_step] = value
            self.request_face(Face.from_array(parameters), i)

    def receive_image(self, image, index=None):
        if index == 'init':
            return
        elif index == 'pre':
            self.__get_parameter(self.__current_step + 1)
            return

        shadows = image

        if index is None:
            self.finish(self.__face)
            return

        self.__errors[index] = self.get_image_deviation(shadows)

        if None in self.__errors:
            return

        errors = array(self.__errors)
        max_error = errors.max()
        if self.__loop >= self.__determined_loops:
            X = exp(- errors / max_error).sum()
            v = rand()
            best_index = -1
            t = 0
            for i, error in enumerate(errors):
                e = exp(error / max_error) / X
                t += e
                if v <= e:
                    best_index = i
                    break
                else:
                    v -= e
            if best_index == -1:
                best_index = len(self.__errors) - 1
        else:
            best_index = argmin(self.__errors)

        self.__parameters[self.__current_step] = self.__values[best_index]
        self.__face = Face.from_array(self.__parameters)
        print('{}.{}: Error of the best is {} ({})'.format(
            self.__loop, self.__current_step, self.__errors[best_index],
            min(self.__errors)))

        if self.__current_step + 1 == self._dimensions \
                and self.__loop + 1 >= self.__max_loops:
            self.request_face(self.__face)
            return
        elif self.__current_step + 1 == self._dimensions:
            self.__current_step = 0
            self.__loop += 1
            self.request_face(self.__face, 'pre')
            return

        self.request_face(self.__face, 'pre')
