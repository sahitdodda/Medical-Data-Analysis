# linked list classssssssss 
class Node: 
    def __init__(self, data=None):
        self.data = data 
        self.next = None

class LL: 
    def __init__(self):
        self.head = None # head of list is none
    def append(self, data):
        new_node = Node(data) # construct node from data 
        if self.head is None: 
            self.head = new_node # if list empty, new node is head 
        else:
            current = self.head
            while current.next: # loop to get to last element of the linked list
                current = current.next  
            current.next = new_node
    def display(self):
        current = self.head
        while current: 
            print(current.data)
            current = current.next # displaying the list, not equiv to concatenation
    def length(self):
        current = self.head
        count = 0
        while current: 
            count += 1
            current = current.next # displaying the list, not equiv to concatenation
        return count
