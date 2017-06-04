def works_fine():
	a = 5
	b = 6
	assert (a + b == 11)
	return None

def throws_an_exception():
	a = 5
	b = 6
	assert (a + b == 10)
	return None

def calling_things():
	works_fine()
	throws_an_exception()

calling_things()
