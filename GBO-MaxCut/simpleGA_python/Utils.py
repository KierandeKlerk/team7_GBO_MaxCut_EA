class ValueToReachFoundException(Exception):
	def __init__(self, individual):            
		super().__init__("Value to reach found")
		self.individual = individual
