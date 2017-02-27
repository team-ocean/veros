import bohrium as bh

@bh.replace_numpy
def run():
	from acc2 import ACC2
	acc = ACC2()
	acc.run(runlen = 3600 * 24 * 360, snapint = bh.inf)

run()
