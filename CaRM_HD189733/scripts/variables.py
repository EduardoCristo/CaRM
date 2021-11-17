def modelvar(model):
    storbitpar = {"model": "", "ld_law": "", "Rstar": None, "steff": None, "sig_steff": None,
                  "slog": None, "sig_slog": None, "sz": None, "sig_sz": None, "tepoch": None,"delta_transit": None, "P": None}
    storbitpar["model"] = model

    Wpar=dict()
    Wguess=dict()
    Wpriors=dict()
    Wsingle=dict()
    Cpar=dict()
    Cguess=dict()
    Cpriors=dict()
    Csingle=dict()

    if storbitpar["model"] == "pyastronomy":
        pyastkeys=["vsys","rp","k","sma","inc","lda","ldc","Vrot","Is","Omega","dT0","sigw","act_slope","ln_a","ln_tau"]
        for key in pyastkeys:
            Wpar[key]=None
            Wguess[key]=None
            Wpriors[key]=[]
            Wsingle[key]=None
            Cpar[key]=None
            Cguess[key]=None
            Cpriors[key]=[]
            Csingle[key]=None


    elif storbitpar["model"] == "pyarome":
        varnames=[r"$V_{sys}$", r"$R_{p}/R_{*}$", r"$K$", r"$a$", r"$i_*$", r"$\lambda$", r"$u_{1}$", r"$u_{2}$",r"$u_{3}$",r"$u_{4}$", r"$\beta_{0}$",
               r"$v\,sin\,i_*$", r"$log(\sigma_{0})$", r"$\zeta$", r"$K_{max}$", r"$\Delta T_{0}$", r"$log( \sigma_W )$",r"$\xi$",r"$log(a)$",r"$log(\tau)$"]
        vardict_plot=dict()
        aromekeys=["vsys","rp","k","sma","inc","lda","ldc1","ldc2","ldc3","ldc4","beta0","Vsini","sigma0","zeta","Kmax","dT0","sigw","act_slope","ln_a","ln_tau"]
        c=0
        for key in aromekeys:
            vardict_plot[key]=varnames[c]
            c+=1

        for key in aromekeys:
            Wpar[key]=None
            Wguess[key]=None
            Wpriors[key]=[]
            Wsingle[key]=None
            Cpar[key]=None
            Cguess[key]=None
            Cpriors[key]=[]
            Csingle[key]=None
        

    else:
        import sys
        sys.exit("No model defined")
    return(storbitpar, Wpar, Wguess, Wpriors, Cpar, Cguess, Cpriors, Wsingle, Csingle,vardict_plot)
