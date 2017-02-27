import bohrium as bh

@bh.replace_numpy
def run():
	from global_one_degree import GlobalOneDegree
	sim = GlobalOneDegree()
	sim.run(runlen = 3600 * 24 * 360, snapint = bh.inf)

run()
