




class Divider:
	def __init__(self, percent):
		self.labels    = open("data/original_data/labels.ndjson", "r")
		self.celebrity = open("data/original_data/celebrity-feeds.ndjson", "r")
		self.percent = percent

	def create_split(self):		
		try:
			train_labels    = open("./data/split_data/training/labels_train.ndjson", "w")
			train_celebrity = open("./data/split_data/training/celebrity_train.ndjson", "w")

			test_labels    = open("data/split_data/test/labels_test.ndjson", "w")
			test_celebrity = open("data/split_data/test/celebrity_test.ndjson", "w")


			nr_0 = 0
			nr_1 = 0
			for index, line in enumerate(self.celebrity):
				if (index/1919)>self.percent :
					nr_1 += 1
					# Test
					test_celebrity.write(line)

				else: 
					# Train
					nr_0 += 1
					train_celebrity.write(line)
			print(f"Celebrity files: \nTrain size:{nr_0} - {int(nr_0*100/1920)}%\tTest size:{nr_1} - {int(nr_1*100/1920)}%")

			nr_0 = 0
			nr_1 = 0
			for index, line in enumerate(self.labels):
				if (index/1919)>self.percent :
					nr_1 += 1
					# Test
					test_labels.write(line)

				else: 
					# Train
					nr_0 += 1
					train_labels.write(line)
			print(f"Labels files: \nTrain size:{nr_0} - {int(nr_0*100/1920)}%\tTest size:{nr_1} - {int(nr_1*100/1920)}%")
			
		except:
			print("Does exist =>", str(pe))
