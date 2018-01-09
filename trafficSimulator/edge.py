#! /users/guruprakash.r/miniconda2/bin/python

from node import Node 


class Edge:
	def __init__(self, node1, node2, width):
		self.node1 = node1
		self.node2 = node2
		self.width = width

	def __str__(self):
		return "Edge: " + str(self.node1) + str(self.node2) + str(self.width) + "\n"

	def __repr__(self):
		return  "Edge: " + str(self.node1) + str(self.node2) + "\n Width: " + str(self.width) + "\n"