import csv
jobs = []
userSkills = []
def getUserSkills():
	global userSkills
	with open("userSkills.csv") as csvFile:
		csvreader = csv.reader(csvFile)
		userSkills = next(csvreader)

def getJobs():
	with open("Jobs.csv") as csvFile:
		csvreader = csv.reader(csvFile)
		for row in csvreader:
			jobs.append(row)
def getMatchedJobs():
	userSkillsSet = set(userSkills)
	matchedJobs = []
	for job in jobs:
		jobSkillSet = set(job[2:])
		if jobSkillSet <= userSkillsSet:
			matchedJobs.append(job[:2])
	return matchedJobs
def getSkillsNeeded():
	userSkillsSet = set(userSkills)
	skillsNeeded = []
	for job in jobs:
		jobSkillSet = set(job[2:])
		difference = jobSkillSet - userSkillsSet
		if difference:
			skillsNeeded.append(job[:2]+ list(difference))
	return skillsNeeded
def main():
	getUserSkills()
	print("User Skills:")
	print(*userSkills,sep=", ")
	getJobs()
	matchedJobs = getMatchedJobs()
	print("\nEligible Jobs:")
	for job in matchedJobs:
		print(*job,sep =", ")
	skillsNeeded = getSkillsNeeded()
	print("\nSkills Needed for Other Jobs:")
	for job in skillsNeeded:
		print(*job,sep =", ")
if __name__== "__main__":
  main()
