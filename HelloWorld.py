class HelloWorld:
    def __init__(self, hello):
        self.hello = hello

    def print_hello(self):
        print('Hello..', self.hello)

seyHello = HelloWorld('World!')
seyHello.print_hello()