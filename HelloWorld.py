class HelloWorld:
    def __init__(self, hello, world):
        self.hello = hello
        self.world = world

    def print_hello(self):
        print('Hello..', self.hello)

    def print_world(self):
        print('World..', self.world)

class NeuralNetwork:
    def __ini__ (self, hidden_layer_size = 100):
        self.hidden_layer_size = hidden_layer_size

seyHello = HelloWorld('Hello!', 'World!')
seyHello.print_hello()
seyHello.print_world()