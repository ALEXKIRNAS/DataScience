def set_trace():
	from IPyton.core.debugger import Pdb
	Pdb(color_scheme = 'Linux').set_trace(sys._getframe().f_back)
	
def debug(f, *args, **kwargs):
	from IPyton.core.debugger import Pdb
	pdb = Pdb(color_scheme = 'Linux')
	return pdb.runcall(f, *args, **kwargs)