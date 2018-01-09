
from node import Node 
from edge import Edge
import cv2 

class RoadNetwork:
		def __init__(self, nodes, edges):
			self.nodes = nodes
			self.edges = edges


		def drawNetwork(self):
			pass

		def isContinuous(self):
			return true

		def __str__(self):
			st =  "Nodes: "
			for node in self.nodes:
				st += str(node)

			for edge in self.edges:
				st += str(edge)
			return st

		def __repr__(self):
			st =  "Nodes: "
			for node in nodes:
				st += str(node)

			for edges in edges:
				st += str(edge)
			return st
