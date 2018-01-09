#! /users/guruprakash.r/miniconda2/bin/python

from node import Node 
from edge import Edge

class EdgeWithLanes(Edge):

	def __init__(self, node1, node2, width, NoOflanes):
		self.node1 = node1
		self.node2 = node2
		self.width = width
		self.lanes = NoOflanes

	def __str__(self):
		return "Edge: " + str(self.node1) + str(self.node2) + str(self.width) + str(self.lanes) +"\n"

	def __repr__(self):
		return  "Edge: " + str(self.node1) + str(self.node2) + "\n Width: " + str(self.width) + "\n"