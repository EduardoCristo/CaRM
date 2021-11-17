import numpy as np
"""
def random_chain(prior_type,prior_interval,positions):
    dlen=len(positions)
    rand_chain=[]
    for k in xrange(dlen):
        pos=positions[k]
        if prior_type[pos]=="U":
            rand_chain+= [np.random.uniform(prior_interval[pos][0],prior_interval[pos][1])]
        elif prior_type[pos]=="G":
            rand_chain+= [np.random.normal(prior_interval[pos][0],prior_interval[pos][1])]
        elif prior_type[pos]=="None":
            import sys
            sys.exit("WARNING: Some parameter is not defined in the constants file!!!")
    return(rand_chain)

def random_chain(prior_type, prior_interval):
    rand_chain = []
    dlen = len(prior_type)
    for k in range(dlen):
        pos = k
        if prior_type[pos] == "U":
            rand_chain += [np.random.uniform(prior_interval[pos]
                                             [0], prior_interval[pos][1])]
        elif prior_type[pos] == "G":
            rand_chain += [np.random.normal(prior_interval[pos]
                                            [0], prior_interval[pos][1])]
        elif prior_type[pos] == "None":
            import sys
            sys.exit(
                "WARNING: Some parameter is not defined in the constants file!!!")
    return(rand_chain)
"""


def random_chain(prior_type, prior_interval,guess,single,nchains,dlen):
    rand_chain=[]
    outdict=[]
    import sys

    for key in prior_type:
        if single[key]==True:
            for i in range(dlen):
                if prior_type[key]=="U":
                    rand_chain+= [np.random.uniform(prior_interval[key][0], prior_interval[key][1], nchains)]
                    outdict+=[key+"_"+str(i)]
                elif prior_type[key]=="G":
                    rand_chain+= [np.random.normal(prior_interval[key][0], prior_interval[key][1], nchains)]
                    outdict+=[key+"_"+str(i)]

        elif single[key]==None:
            if prior_type[key]=="U":
                rand_chain+= [np.random.uniform(prior_interval[key][0], prior_interval[key][1], nchains)]
                outdict+=[key]
            elif prior_type[key]=="G":
                rand_chain+= [np.random.normal(prior_interval[key][0], prior_interval[key][1], nchains)]
                outdict+=[key]
    print(outdict)
    return(np.array(rand_chain).T,outdict)

    
