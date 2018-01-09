#! /users/guruprakash.r/miniconda2/bin/python

class Node:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __str__(self):
		return "Node: " + str(self.x) +  " " + str(self.y) + "\n"

	def __repr__(self):
		return "Node: " + str(self.x) +  " " + str(self.y) + "\n"
