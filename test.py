class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [None] * capacity
        self.size = 0
        self.start = 0

    def __len__(self):
        return self.size

    def append(self, item):
        if self.size < self.capacity:
            self.data[(self.start + self.size) % self.capacity] = item
            self.size += 1
        else:
            self.data[self.start] = item
            self.start = (self.start + 1) % self.capacity

    def get(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        return self.data[(self.start + index) % self.capacity]

def test1():

    # Example usage
    buffer = RingBuffer(5)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    buffer.append(5)
    buffer.append(6)  # Overwrites the oldest value (1)
    print(buffer.get(0))  # Output: 2
    print(buffer.get(1))  # Output: 3
    print(buffer.get(2))  # Output: 4
    print(buffer.get(3))  # Output: 5
    print(buffer.get(4))  # Output: 6


from collections import deque
def test2():
    # Example usage
    buffer = deque(maxlen=5)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)
    buffer.append(4)
    buffer.append(5)
    buffer.append(6)  # Overwrites the oldest value (1)
    print(buffer[0])  # Output: 2
    print(buffer[1])  # Output: 3
    print(buffer[2])  # Output: 4
    print(buffer[3])  # Output: 5
    print(buffer[4])  # Output: 6

def test3():
    buffer = deque(maxlen=5)
    i = 1
    while i <= 6:
        buffer.append(i)
        i = i+1
    
    print(buffer[0])  # Output: 2
    print(buffer[1])  # Output: 3
    print(buffer[2])  # Output: 4
    print(buffer[3])  # Output: 5
    print(buffer[4])  # Output: 6

test3()