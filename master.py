import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import py3nj
import time
from numba import jit,prange
import warnings
import os
from matplotlib import gridspec

LMAX = 0
NSIDE = 0

write_time = time.time()

##### Function to start wigner coefficient calculation ######
def mydrc3jj(two_l2, two_l3, two_m2, two_m3):
    """ scalar version of drc3jj """
    l1max = (two_l2 + two_l3)
    l1min = max(np.abs(two_l2 - two_l3), np.abs(two_m2 + two_m3))
    ndim = int((l1max - l1min + 2) / 2)

    l1min, l1max, thrcof, ier = py3nj. _wigner.drc3jj_int(
        two_l2, two_l3, two_m2, two_m3, ndim)
    if ier == 1:
        raise ValueError('Either L2.LT.ABS(M2) or L3.LT.ABS(M3).')
    elif ier == 2:
        raise ValueError('Either L2+ABS(M2) or L3+ABS(M3) non-integer.')
    elif ier == 3:
        raise ValueError('L1MAX-L1MIN not an integer.')
    elif ier == 4:
        raise ValueError('L1MAX less than L1MIN.')
    elif ier == 5:
        raise ValueError('NDIM less than L1MAX-L1MIN+1.')

    return thrcof



##### Function to calculate the symmetric part of coupling matrix for TT spectra ######
@jit(nopython=False,parallel=True)
def N_l1_l2_TT(l1,l2,LMAX,window_cl):
    s = 0
    l3max = l1+l2
    l3min = abs(l1-l2)
    ndim = l3max - l3min
    coeff=mydrc3jj(2*l1,2*l2,0,0)**2
    i=0
    for l3 in range(l3max,l3min-1,-2):
        if l3 < LMAX+1:
            s=s+((2*l3)+1)*window_cl[l3]*coeff[ndim-i]
        i = i+2
    return s

##### Function to calculate the symmetric part of coupling matrix for TE spectra #####
@jit(nopython=False,parallel=True)
def N_l1_l2_TE(l1,l2,LMAX,window_cl):
    sum_te = 0
    l3max = l1+l2
    l3min = abs(l1-l2)
    ndim = l3max - l3min
    coeff1=mydrc3jj(2*l1,2*l2,0,0)
    coeff2=mydrc3jj(2*l1,2*l2,4,-4)
    i=0
    for l3 in range(l3max,l3min-1,-2):
        if l3 < LMAX+1:
            sum_te=sum_te+((2*l3)+1)*window_cl[l3]*coeff1[ndim-i]*coeff2[ndim-i]
        i = i+2
    return sum_te



##### Function to calculate the symmetric part of coupling matrix for EE spectra and BB spectra #####
@jit(nopython=False,parallel=True)
def N_l1_l2_EE_BB(l1,l2,LMAX,window_cl):
    sum_ee_bb = 0
    sum_ee_ee = 0
    l3max = l1+l2
    l3min = abs(l1-l2)
    ndim = l3max - l3min
    coeff1=mydrc3jj(2*l1,2*l2,0,0)
    coeff2=mydrc3jj(2*l1,2*l2,4,-4)
    i=0
    for l3 in range(l3max,l3min-1,-2):
        if l3 < LMAX+1:
            sum_ee_ee=sum_ee_ee+((2*l3)+1)*window_cl[l3]*coeff1[ndim-i]*coeff2[ndim-i]
        i=i+2
    i=0
    for l3 in range(l3max,l3min-1):
        if l3 < LMAX+1:
            sum_ee_bb=sum_ee_bb+(2*(l3)+1)*window_cl[l3]*coeff2[ndim-i]**2
        i = i+2
    return sum_ee_ee,sum_ee_bb


##### Function to generate theory plots #####
def PLOT_THEORY(cl,master_cl,full_sky_cl,masked_cl,LMAX,no):
    l=np.arange(2,LMAX)
    fig = plt.figure(figsize=(8,6))
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4,1], hspace=0.01)
    ax0 = fig.add_subplot(spec[0])
    #ax0.grid()
    ax0.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax0.set_ylabel("${l(l+1)C_l}/{2\pi}$", fontsize=11, fontweight='bold')


    ax0.get_yaxis().get_major_formatter().set_scientific(False)
    from matplotlib.ticker import FormatStrFormatter
    #ax0.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax0.yaxis.set_label_position("right")
    ax0.set_title('Plot after '+str(no)+' realisation', fontsize=12,fontweight='bold')

    if choice ==2 or choice == 3:
        delta_master = master_cl-cl[2:LMAX+1]
    else:    
        delta_master = master_cl-cl[0:LMAX+1]
    
    delta_fs = full_sky_cl-cl[0:LMAX+1]

    ax1 = fig.add_subplot(spec[1])
    ax1.grid()
    ax1.set_ylabel(r'$\Delta C_l$', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Multipole moment l', fontsize=11, fontweight='bold')
    if choice == 2 or choice == 3:
        ax0.plot(l,l*(l+1)*master_cl[l-2]/(2*np.pi),'g')
        ax0.plot(l,l*(l+1)*masked_cl[l-2]/(2*np.pi),'y')
        ax1.plot(l,l*(l+1)*delta_master[l-2]/(2*np.pi),'g')
    else:
        ax0.plot(l,l*(l+1)*master_cl[l]/(2*np.pi),'g')
        ax0.plot(l,l*(l+1)*masked_cl[l]/(2*np.pi),'y')
        ax1.plot(l,l*(l+1)*delta_master[l]/(2*np.pi),'g')
    #ax0.plot(l,l*(l+1)*full_sky_cl[l]/(2*np.pi),'r')
    ax0.plot(l,l*(l+1)*cl[l]/(2*np.pi),'b')
    ax0.legend(["MASTER $C_l$ ","Masked $C_l$","full sky $C_l$","theoretical $C_l$" ], loc ="upper right")
    ax1.plot(l,l*(l+1)*delta_fs[l]/(2*np.pi),'r')

    
    ax1.yaxis.set_label_position("left")
    ax0.yaxis.set_label_position("left")
    plt.savefig("Plot_"+str(write_time)+"_"+str(LMAX)+".pdf", format='pdf')
    



##### Function to generate plots with input maps #####    
def PLOT_MAPS(cl,master_cl,full_sky_cl,masked_cl,LMAX,no,NSIDE):
    l=np.arange(2,LMAX)
    fig = plt.figure(figsize=(8,6))
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4,1], hspace=0.01)
    ax0 = fig.add_subplot(spec[0])
    #ax0.grid()
    ax0.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax0.set_ylabel("${l(l+1)C_l}/{2\pi}$", fontsize=11, fontweight='bold')


    ax0.get_yaxis().get_major_formatter().set_scientific(False)
    from matplotlib.ticker import FormatStrFormatter
    #ax0.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax0.yaxis.set_label_position("right")
    ax0.set_title('Plot after '+str(no)+' realisation', fontsize=12,fontweight='bold')

    bl = hp.gauss_beam(5.0/60*np.pi/180, lmax=LMAX, pol=False)
    pl = hp.pixwin(NSIDE) 
    
    delta_master = master_cl*10**12-cl[0:LMAX+1]
    delta_fs = full_sky_cl*10**12-cl[0:LMAX+1]
    
    

    ax1 = fig.add_subplot(spec[1])
    ax1.grid()
    ax1.set_ylabel(r'$\Delta C_l$', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Multipole moment l', fontsize=11, fontweight='bold')
    ax0.plot(l,l*(l+1)*master_cl[l]*10**12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'g')
    ax0.plot(l,l*(l+1)*full_sky_cl[l]*10**12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'r')
    ax0.plot(l,l*(l+1)*masked_cl[l]*10**12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'y')
    ax0.plot(l,l*(l+1)*cl[l]/(2*np.pi),'b')
    ax0.legend(["MASTER $C_l$ ","full sky $C_l$","Masked $C_l$","theoretical $C_l$" ], loc ="upper right")
    ax1.plot(l,l*(l+1)*delta_master[l]/(2*np.pi),'g')
    ax1.plot(l,l*(l+1)*delta_fs[l]/(2*np.pi),'r')
    
    ax1.yaxis.set_label_position("left")
    ax0.yaxis.set_label_position("left")
    plt.savefig("Plot_"+str(write_time)+"_"+str(LMAX)+".pdf", format='pdf')
    
    
    
##### Function to copy all plots and files into a directory #####    
def copy(cl,window_cl,full_sky_cl,masked_cl,master_cl,LMAX,c,NSIDE):
    os.system("mkdir fits_"+str(write_time)+"_files")
    if c == 1:
        full_sky_map =hp.synfast(cl,NSIDE,lmax=LMAX,alm=False,pol=False,pixwin=False,fwhm=0.0,new=True)
        hp.write_map("full_sky_unmasked_"+str(write_time)+"_.fits",full_sky_map)
        os.system("mv full_sky_unmasked_"+str(write_time)+"_.fits fits_"+str(write_time)+"_files")
    hp.write_cl("Theoretical_APS.fits", cl,overwrite = True)
    hp.write_cl("Window_APS.fits", window_cl,overwrite = True)
    hp.write_cl("Full_sky_APS.fits", full_sky_cl,overwrite= True)
    hp.write_cl("Masked_map_APS.fits", masked_cl,overwrite = True)
    hp.write_cl("Master_APS.fits", master_cl,overwrite = True)
    os.system("mv Theoretical_APS.fits Window_APS.fits Full_sky_APS.fits Masked_map_APS.fits Master_APS.fits Plot_"+str(write_time)+"_"+str(LMAX)+".pdf fits_"+str(write_time)+"_files")



#### Function to calculate TT spectra #####
def MASTER_TT():
    def theory_cl():
        #####Taking inputs#####
        cl_file = input("Enter the theory C_l file:")
        mask_file = input("Enter the name of the mask file:")
        warnings.filterwarnings('ignore')
        mask = hp.read_map(mask_file)
        NSIDE=hp.get_nside(mask)
        LMAX=int(input("Enter LMAX:"))
        no = int(input("Enter no.of.realisations:"))
        print("This might take a while. Please be patient!!!")
        #####Done with the inputs#####
        cl = hp.read_cl(cl_file)[0]
        if LMAX >= 3*NSIDE-1:
            print("LMAX you entered is more than 3*NSIDE-1. LMAX will be overwritten as 3*NSIDE-1!!!!")
            LMAX = 3*NSIDE-1
        if LMAX>len(cl):
            print("LMAX you entered is more than the length of C_l file!!!")
            exit()
        
        

        window_cl = hp.anafast(mask,lmax = LMAX)


        M = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        N = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

        s=0
        start_time = time.time()
        for l1 in range(2,LMAX+1):
            for l2 in range(l1,LMAX+1):
                N[l1][l2] = N_l1_l2_TT(l1,l2,LMAX,window_cl)
                N[l2][l1] = N[l1][l2]
        print("Time taken for the program to run is:",(time.time()-start_time)/60 ," minutes")
        for l1 in range(2,LMAX+1):
            for l2 in range(2,LMAX+1):
                M[l1][l2] = ((2*l2+1)/(4*np.pi))* N[l1][l2]

        M_sub = [ [ 1 for i in range(LMAX-1) ] for j in range(LMAX-1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_sub[l1][l2] = M[l1+2][l2+2]
        M_inv_sub = np.linalg.inv(M_sub)
        M_inv = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_inv[2+l1][2+l2] = M_inv_sub[l1][l2]

        cl=hp.read_cl(cl_file)[0]
        full_sky_cl = np.zeros(LMAX+1)
        masked_cl = np.zeros(LMAX+1)


        def temp_sum(l1,masked_cl):
            s = 0
            for l2 in range(2,LMAX+1):
                s = s + M_inv[l1][l2]*masked_cl[l2]
            return s

        master_cl = np.zeros(LMAX+1)

        def get_master(masked_cl):
            l1  = 2
            for i in range(2,LMAX+1):
                master_cl[i] = temp_sum(l1,masked_cl)
                l1+=1
            return master_cl

        for i in range(no):
            full_sky_map =hp.synfast(cl,NSIDE,lmax=LMAX,alm=False,pol=False,pixwin=False,fwhm=0.0,new=True)
            hp.write_map("full_sky_unmasked_"+str(write_time)+"_"+str(no)+".fits",full_sky_map)
            full_sky_cl =full_sky_cl + hp.anafast(full_sky_map, lmax=LMAX)/no
            masked_map = full_sky_map*mask
            masked_cl = masked_cl + hp.anafast(masked_map,lmax=LMAX)/no    

        master_cl = get_master(masked_cl)


        PLOT_THEORY(cl,master_cl,full_sky_cl,masked_cl,LMAX,no)
        copy(cl,window_cl,full_sky_cl,masked_cl,master_cl,LMAX,c,NSIDE)
        


    def input_maps():
        #####Taking inputs#####
        cl_file = input("Provide a theory C_l file for reference:")
        map1_file = input("Enter the name of the first map:")
        sc = input("Do you want to enter the second map(y/n):")
        if sc=='y':
            map2_file = input("Enter the name of the second map:")
            map2 = hp.read_map(map2_file)
        mask_file = input("Enter the name of the mask file:")
        LMAX=int(input("Enter LMAX:"))
        no = int(input("Enter no.of.realisations:"))
        #####Done with the inputs#####
        mask = hp.read_map(mask_file)
        map1 = hp.read_map(map1_file)
        if hp.get_nside(mask)!=hp.get_nside(map1):
            print("Nside of mask and map are not same")
        NSIDE=hp.get_nside(map1)
        if LMAX >= 3*NSIDE-1:
            print("LMAX you entered is more than 3*NSIDE-1!!!!")
            exit()
        if LMAX%2 == 0:
            LMAX = LMAX+1
        warnings.filterwarnings('ignore')
        print("This might take a while. Please be patient!!!")

        window_cl = hp.anafast(mask,lmax = LMAX)



        M = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        N = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

        s=0
        start_time = time.time()
        for l1 in range(2,LMAX+1):
            for l2 in range(l1,LMAX+1):
                N[l1][l2] = N_l1_l2_TT(l1,l2,LMAX,window_cl)
                N[l2][l1] = N[l1][l2]
        print("Time taken for the program to run is:",(time.time()-start_time)/60 ," minutes")
        for l1 in range(2,LMAX+1):
            for l2 in range(2,LMAX+1):
                M[l1][l2] = ((2*l2+1)/(4*np.pi))* N[l1][l2]

        M_sub = [ [ 1 for i in range(LMAX-1) ] for j in range(LMAX-1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_sub[l1][l2] = M[l1+2][l2+2]
        M_inv_sub = np.linalg.inv(M_sub)
        M_inv = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_inv[2+l1][2+l2] = M_inv_sub[l1][l2]



        cl=hp.read_cl(cl_file)[0]
        full_sky_cl = np.zeros(LMAX+1)
        masked_cl = np.zeros(LMAX+1)


        def temp_sum(l1,masked_cl):
            s = 0
            for l2 in range(2,LMAX+1):
                s = s + M_inv[l1][l2]*masked_cl[l2]
            return s

        master_cl = np.zeros(LMAX+1)

        def get_master(masked_cl):
            l1  = 2
            for i in range(2,LMAX+1): 
                master_cl[i] = temp_sum(l1,masked_cl)
                l1+=1
            return master_cl




        if sc=='y':
            for i in range(no):
                full_sky_cl = full_sky_cl + hp.anafast(map1,map2,lmax = LMAX)/no
                masked_cl = masked_cl + hp.anafast(map1*mask,map2*mask,lmax=LMAX)/no
        else:
            for i in range(no):
                full_sky_cl =full_sky_cl + hp.anafast(map1, lmax=LMAX)/no
                masked_map = map1*mask
                masked_cl = masked_cl + hp.anafast(masked_map,lmax=LMAX)/no    

        master_cl = get_master(masked_cl)


        bl = hp.gauss_beam(5.0/60*np.pi/180, lmax=LMAX, pol=False)
        pl = hp.pixwin(NSIDE)   
        l = np.arange(2,LMAX)
        l_arr = np.arange(0,LMAX+1)

        
        PLOT_MAPS(cl,master_cl,full_sky_cl,masked_cl,LMAX,no,NSIDE)
        copy(cl,window_cl,full_sky_cl,masked_cl,master_cl,LMAX,c,NSIDE)
        



    c = int(input("Enter 1 if you want to generate Master spectrum using simulated one or enter 2 if you want to generate using input maps:"))


    if c ==1:
        print("You have selected Theory spectra!!!")
        theory_cl()
    if c ==2:
        print("You have selected Input maps!!!")
        input_maps()
    print("Goodbye!!")

    

#### Function to calculate TE spectra #####
def MASTER_TE():    
    def theory_cl():
        #####Taking inputs#####
        warnings.filterwarnings('ignore')
        cl_file = input("Enter the theory C_l file:")
        mask_file_t = input("Enter the name of the temperature mask file:")
        mask_file_p = input("Enter the name of the polarization mask file:")
        mask_t = hp.read_map(mask_file_t)
        mask_p = hp.read_map(mask_file_p)
        NSIDE = hp.get_nside(mask_t)
        if hp.get_nside(mask_p) == NSIDE:
            print("The NSIDE of temperature and polarization masks are different!!")
            exit()
        LMAX=int(input("Enter LMAX:"))
        no = int(input("Enter no.of.realisations:"))
        print("This might take a while. Please be patient!!!")
        p_time=time.time()
        #####Done with the inputs#####
        test_cl = hp.read_cl(cl_file)[3]
        if LMAX >= 3*NSIDE-1:
            print("LMAX you entered is more than 3*NSIDE-1. LMAX will be overwritten as 3*NSIDE-1!!!!")
            LMAX = 3*NSIDE-1
        if LMAX>len(test_cl):
            print("LMAX you entered is more than the length of C_l file!!!")
            exit()
        warnings.filterwarnings('ignore')


        window_cl = hp.anafast(mask_t,mask_p,lmax = LMAX)




        M = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        N = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

        s=0
        start_time = time.time()
        for l1 in range(2,LMAX+1):
            for l2 in range(l1,LMAX+1):
                N[l1][l2] = N_l1_l2_TE(l1,l2,LMAX,window_cl)
                N[l2][l1] = N[l1][l2]
        print("Time taken for the matrix to be computed is:",(time.time()-start_time)/60 ," minutes")
        for l1 in range(2,LMAX+1):
            for l2 in range(2,LMAX+1):
                M[l1][l2] = ((2*l2+1)/(4*np.pi))* N[l1][l2]

        M_sub = [ [ 1 for i in range(LMAX-1) ] for j in range(LMAX-1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_sub[l1][l2] = M[l1+2][l2+2]
        M_inv_sub = np.linalg.inv(M_sub)
        M_inv = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_inv[2+l1][2+l2] = M_inv_sub[l1][l2]

        cl=hp.read_cl(cl_file)
        full_sky_cl_te = np.zeros(LMAX+1)
        masked_cl_te = np.zeros(LMAX+1)


        def temp_sum(l1,masked_cl):
            s = 0
            for l2 in range(2,LMAX+1):
                s = s + M_inv[l1][l2]*masked_cl[l2]
            return s

        master_cl_te = np.zeros(LMAX+1)
        master_cl = np.zeros(LMAX+1)

        def get_master(masked_cl):
            l1  = 2
            for i in range(2,LMAX+1):
                master_cl[i] = temp_sum(l1,masked_cl)
                l1+=1
            return master_cl


        write_time = time.time()
        for i in range(no):
            full_sky_map =hp.synfast(cl,NSIDE,lmax=LMAX,alm=False,pol=False,pixwin=False,fwhm=0.0,new=True)
            masked_map = [ [ 1 for i in range(12*NSIDE**2) ] for j in range(3)]
            masked_map[0] = full_sky_map[0]*mask_t
            masked_map[1] = full_sky_map[1]*mask_p
            masked_map[2] = full_sky_map[2]*mask_p
            full_sky_cl_te =full_sky_cl_te + hp.anafast(full_sky_map[0],full_sky_map[1], lmax=LMAX)/no
            masked_cl_te = masked_cl_te + hp.anafast(masked_map[0],masked_map[1],lmax=LMAX)/no

        master_cl_te = get_master(masked_cl_te)
        
        

        
        print("Time taken for the program to run is:",(time.time()-p_time)/60 ," minutes")
        
        PLOT_THEORY(cl[3],master_cl_te,full_sky_cl_te,masked_cl_te,LMAX,no)
        copy(cl[3],window_cl,full_sky_cl_te,masked_cl_te,master_cl_te,LMAX,c,NSIDE)
        
    
    
    def input_maps():
        #####Taking inputs#####
        warnings.filterwarnings('ignore')
        cl_file = input("Provide a theory C_l file for reference:")
        map1_file = input("Enter the name of the first map:")
        sc = input("Do you want to enter the second map(y/n):")
        if sc=='y':
            map2_file = input("Enter the name of the second map:")
            map2 = hp.read_map(map2_file,field = (0,1,2))
        mask_file_t = input("Enter the name of the temperature mask file:")
        mask_file_p = input("Enter the name of the polarization mask file:")
        LMAX=int(input("Enter LMAX:"))
        no = int(input("Enter no.of.realisations:"))
        print("This might take a while. Please be patient!!!")
        #####Done with the inputs#####
        p_time = time.time()
        mask_t = hp.read_map(mask_file_t)
        mask_p = hp.read_map(mask_file_p)
        map1 = hp.read_map(map1_file,field = (0,1,2))
        if sc == 'y':
            if hp.get_nside(map1)!=hp.get_nside(map2):
                print("Nside of both the maps are different!!!")
                exit()
        if hp.get_nside(mask_t)!=hp.get_nside(map1) or hp.get_nside(mask_p)!=hp.get_nside(map1):
            print("Nside of mask and map are different!!!")
            exit()
        NSIDE=hp.get_nside(map1)
        if LMAX >= 3*NSIDE-1:
            print("LMAX you entered is more than 3*NSIDE-1!!!!")
            exit()


        window_cl = hp.anafast(mask_t,mask_p,lmax = LMAX)

        M = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        N = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

        s=0
        start_time = time.time()
        for l1 in range(2,LMAX+1):
            for l2 in range(l1,LMAX+1):
                N[l1][l2] = N_l1_l2_TE(l1,l2,LMAX,window_cl)
                N[l2][l1] = N[l1][l2]
        print("Time taken for the Matrix to be computed is:",(time.time()-start_time)/60 ," minutes")
        for l1 in range(2,LMAX+1):
            for l2 in range(2,LMAX+1):
                M[l1][l2] = ((2*l2+1)/(4*np.pi))* N[l1][l2]

        M_sub = [ [ 1 for i in range(LMAX-1) ] for j in range(LMAX-1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_sub[l1][l2] = M[l1+2][l2+2]
        M_inv_sub = np.linalg.inv(M_sub)
        M_inv = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_inv[2+l1][2+l2] = M_inv_sub[l1][l2]



        cl=hp.read_cl(cl_file)
        full_sky_cl = np.zeros(LMAX+1)
        masked_cl = np.zeros(LMAX+1)


        def temp_sum(l1,masked_cl):
            s = 0
            for l2 in range(2,LMAX+1):
                s = s + M_inv[l1][l2]*masked_cl[l2]
            return s

        master_cl = np.zeros(LMAX+1)

        def get_master(masked_cl):
            l1  = 2
            for i in range(2,LMAX+1): 
                master_cl[i] = temp_sum(l1,masked_cl)
                l1+=1
            return master_cl




        if sc=='y':
            for i in range(no):
                full_sky_cl = full_sky_cl + hp.anafast(map1,map2,lmax = LMAX)/no
                masked_map1 = [ [ 1 for i in range(12*NSIDE**2) ] for j in range(3)]
                masked_map1[0] = map1[0]*mask_t
                masked_map1[1] = map1[1]*mask_p
                masked_map1[2] = map1[2]*mask_p
                masked_map2 = [ [ 1 for i in range(12*NSIDE**2) ] for j in range(3)]
                masked_map2[0] = map2[0]*mask_t
                masked_map2[1] = map2[1]*mask_p
                masked_map2[2] = map2[2]*mask_p
                masked_cl = masked_cl + hp.anafast(masked_map1,masked_map2,lmax=LMAX)/no
        else:
            for i in range(no):
                full_sky_cl =full_sky_cl + hp.anafast(map1, lmax=LMAX)/no
                masked_map1 = [ [ 1 for i in range(12*NSIDE**2) ] for j in range(3)]
                masked_map1[0] = map1[0]*mask_t
                masked_map1[1] = map1[1]*mask_p
                masked_map1[2] = map1[2]*mask_p
                masked_cl = masked_cl + hp.anafast(masked_map1,lmax=LMAX)/no    

        master_cl = get_master(masked_cl[3])

        print("Time taken for the program to run is:",(time.time()-p_time)/60 ," minutes")
        PLOT_MAPS(cl[3],master_cl,full_sky_cl[3],masked_cl[3],LMAX,no,NSIDE)
        copy(cl[3],window_cl,full_sky_cl[3],masked_cl[3],master_cl,LMAX,c,NSIDE)


    c = int(input("Enter 1 if you want to generate Master spectrum using simulated one or enter 2 if you want to generate using input maps:"))


    if c ==1:
        print("You have selected Theory spectra!!!")
        theory_cl()
    if c ==2:
        print("You have selected Input maps!!!")
        input_maps()
    print("Goodbye!!")



#### Function to calculate EE and BB spectra #####
def MASTER_EE_BB():
    def theory_cl():
        #####Taking inputs#####
        cl_file = input("Enter the theory C_l file:")
        mask_file = input("Enter the name of the polarization mask file:")
        warnings.filterwarnings('ignore')
        mask = hp.read_map(mask_file)
        NSIDE=hp.get_nside(mask)
        LMAX=int(input("Enter LMAX:"))
        no = int(input("Enter no.of.realisations:"))
        print("This might take a while. Please be patient!!!")
        #####Done with the inputs#####
        test_cl = hp.read_cl(cl_file)[2]
        if LMAX >= 3*NSIDE-1:
            print("LMAX you entered is more than 3*NSIDE-1. LMAX will be overwritten as 3*NSIDE-1!!!!")
            LMAX = 3*NSIDE-1
        if LMAX>len(test_cl):
            print("LMAX you entered is more than the length of C_l file!!!")
            exit()
        
        

        window_cl = hp.anafast(mask,lmax = LMAX)
        
        
        M1 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        N1 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

        M2 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        N2 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

        s=0
        start_time = time.time()



        for l1 in range(2,LMAX+1):
            for l2 in range(l1,LMAX+1):
                N1[l1][l2] = N_l1_l2_EE_BB(l1,l2,LMAX,window_cl)[0]
                N2[l1][l2] = -N_l1_l2_EE_BB(l1,l2,LMAX,window_cl)[1]
                N1[l2][l1] = N1[l1][l2]
                N2[l2][l1] = N2[l1][l2]
        print("Time taken for the matrix to be computed is:",(time.time()-start_time)/60 ," minutes")


        for l1 in range(2,LMAX+1):
            for l2 in range(2,LMAX+1):
                M1[l1][l2] = ((2*l2+1)/(4*np.pi))* N1[l1][l2]
                M2[l1][l2] = ((2*l2+1)/(4*np.pi))* N2[l1][l2]


        M1_sub = [ [ 1 for i in range(LMAX-1) ] for j in range(LMAX-1) ]
        M2_sub = [ [ 1 for i in range(LMAX-1) ] for j in range(LMAX-1) ]
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M1_sub[l1][l2] = M1[l1+2][l2+2]
                M2_sub[l1][l2] = M2[l1+2][l2+2]



        M_sub = [ [ 1 for i in range(2*(LMAX-1)) ] for j in range(2*(LMAX-1)) ]                            
        for l1 in range(LMAX-1):
            for l2 in range(LMAX-1):
                M_sub[l1][l2] = M1_sub[l1][l2]
                M_sub[l1][l2+LMAX-1] = M2_sub[l1][l2]
                M_sub[l1+LMAX-1][l2] = M2_sub[l1][l2]
                M_sub[l1+LMAX-1][l2+LMAX-1] = M1_sub[l1][l2]

        M_inv = np.linalg.inv(M_sub)

        cl=hp.read_cl(cl_file)
        full_sky_cl_ee = np.zeros(LMAX+1)
        full_sky_cl_bb = np.zeros(LMAX+1)
        masked_cl = np.zeros(2*(LMAX-1))

        masked_cl_ee = np.zeros(LMAX+1)
        masked_cl_bb = np.zeros(LMAX+1)

        def temp_sum(l1,masked_cl):
            s = 0
            for l2 in range(0,2*(LMAX-1)):
                s = s + M_inv[l1][l2]*masked_cl[l2]
            return s

        master_cl = np.zeros(2*(LMAX-1))

        def get_master(masked_cl):
            l1  = 0
            for i in range(0,2*(LMAX-1)):
                master_cl[i] = temp_sum(l1,masked_cl)
                l1+=1
            return master_cl


        for i in range(no):
            full_sky_map =hp.synfast(cl,NSIDE,lmax=LMAX,alm=False,pol=False,pixwin=False,fwhm=0.0,new=True)
            masked_map = [ [ 1 for i in range(12*NSIDE**2) ] for j in range(3)]
            masked_map[1] = full_sky_map[1]*mask
            masked_map[2] = full_sky_map[2]*mask
            full_sky_cl_ee = full_sky_cl_ee + hp.anafast(full_sky_map[1], lmax=LMAX)/no
            full_sky_cl_bb =full_sky_cl_bb + hp.anafast(full_sky_map[2], lmax=LMAX)/no
            masked_cl_ee = masked_cl_ee + hp.anafast(masked_map[1],lmax=LMAX)/no
            masked_cl_bb = masked_cl_bb + hp.anafast(masked_map[2],lmax=LMAX)/no

        masked_cl_e = masked_cl_ee[2:]
        masked_cl_b = masked_cl_bb[2:]

        masked_cl = np.concatenate([masked_cl_e,masked_cl_b])
        master_cl = get_master(masked_cl)


        master_cl_ee = master_cl[:len(master_cl)//2]
        master_cl_bb = master_cl[len(master_cl)//2:]
        
        
        if choice == 2:
            PLOT_THEORY(cl[1],master_cl_ee,full_sky_cl_ee,masked_cl_e,LMAX,no)
            copy(cl[1],window_cl,full_sky_cl_ee,masked_cl_e,master_cl_ee,LMAX,c,NSIDE)
        if choice == 3:
            PLOT_THEORY(cl[2],master_cl_bb,full_sky_cl_bb,masked_cl_b,LMAX,no)
            copy(cl[2],window_cl,full_sky_cl_bb,masked_cl_b,master_cl_bb,LMAX,c,NSIDE)
            
            
    
    c = int(input("Enter 1 if you want to generate Master spectrum using simulated one or enter 2 if you want to generate using input maps (Under development):"))


    if c ==1:
        print("You have selected Theory spectra!!!")
        theory_cl()
    if c ==2:
        print("You have selected Input maps!!!")
        print("Under development")
    print("Goodbye!!")
            
    
    
    
print("This code plots MASTER spectrum of theoretical simulations as well as PLANCK input maps.")
print("Choose one of the following options:\n")
print("1. TT spectrum\n2. EE spectrum\n3. BB spectrum\n4. TE spectrum\n")

choice = int(input("Enter desired option: "))


if choice == 1:
    MASTER_TT()
elif choice == 2:
    MASTER_EE_BB()
elif choice == 3:
    MASTER_EE_BB()
elif choice == 4:
    MASTER_TE()
else:
    print("Invalid choice!!!")
    exit()







