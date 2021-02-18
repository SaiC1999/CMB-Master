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


@jit(nopython=False,parallel=True)
def N_l1_l2(l1,l2,LMAX,window_cl):
    sum_tt = 0
    sum_te = 0
    sum_ee_ee = 0
    sum_ee_bb = 0
    sum_eb = 0
    l3max = l1+l2
    l3min = abs(l1-l2)
    ndim = l3max - l3min
    coeff1=mydrc3jj(2*l1,2*l2,0,0)
    coeff2=mydrc3jj(2*l1,2*l2,4,-4)
    if choice == 1:
        i=0
        for l3 in range(l3max,l3min-1,-2):
            if l3 < LMAX+1:
                sum_tt=sum_tt+((2*l3)+1)*window_cl[l3]*coeff1[ndim-i]**2
            i = i+2
        return sum_tt
    elif choice == 2 or choice == 3:
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
    elif choice == 4 or choice == 5:
        i=0
        for l3 in range(l3max,l3min-1,-2):
            if l3 < LMAX+1:
                sum_te=sum_te+((2*l3)+1)*window_cl[l3]*coeff1[ndim-i]*coeff2[ndim-i]
            i = i+2
        return sum_te
    elif choice == 6:
        i=0
        for l3 in range(l3max,l3min-1,-1):
            if l3 < LMAX+1:
                sum_eb=sum_bb+((2*l3)+1)*window_cl[l3]*coeff2[ndim-i]**2
            i = i+1
        return sum_bb
    

    
def PLOT_FUNC(cl,master_cl,full_sky_cl,masked_cl,LMAX,no,NSIDE,c_t_i):
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
    if c_t_i == 1:
        if choice ==2 or choice == 3:
            delta_master = master_cl-cl[2:LMAX+1]
        else:    
            delta_master = master_cl-cl[0:LMAX+1]

        delta_fs = full_sky_cl-cl[0:LMAX+1]

        ax1 = fig.add_subplot(spec[1])
        ax1.grid()
        ax1.set_ylabel(r'${l(l+1)\Delta C_l}/{2\pi}$', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Multipole moment l', fontsize=11, fontweight='bold')
        if choice == 2 or choice == 3:
            ax0.plot(l,l*(l+1)*master_cl[l-2]/(2*np.pi),'g')
            ax0.plot(l,l*(l+1)*masked_cl[l-2]/(2*np.pi),'y')
            ax1.plot(l,l*(l+1)*delta_master[l-2]/(2*np.pi),'g')
        else:
            ax0.plot(l,l*(l+1)*master_cl[l]/(2*np.pi),'g')
            ax0.plot(l,l*(l+1)*masked_cl[l]/(2*np.pi),'y')
            ax1.plot(l,l*(l+1)*delta_master[l]/(2*np.pi),'g')
        ax0.plot(l,l*(l+1)*full_sky_cl[l]/(2*np.pi),'r')
        ax0.plot(l,l*(l+1)*cl[l]/(2*np.pi),'b')
        ax0.legend(["MASTER $C_l$ ","Masked $C_l$","full sky $C_l$","theoretical $C_l$" ], loc ="upper right")
        ax1.plot(l,l*(l+1)*delta_fs[l]/(2*np.pi),'r')


        ax1.yaxis.set_label_position("left")
        ax0.yaxis.set_label_position("left")
        plt.savefig("Plot_"+str(write_time)+"_"+str(LMAX)+".pdf", format='pdf')
    else:
        bl = hp.gauss_beam(5.0/60*np.pi/180, lmax=LMAX, pol=False)
        pl = hp.pixwin(NSIDE) 

        if choice ==2 or choice == 3:
            delta_master = master_cl*1e12-cl[2:LMAX+1]
        else:    
            delta_master = master_cl*1e12-cl[0:LMAX+1]

        delta_fs = full_sky_cl*1e12-cl[0:LMAX+1]



        ax1 = fig.add_subplot(spec[1])
        ax1.grid()
        ax1.set_ylabel(r'$\Delta C_l$', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Multipole moment l', fontsize=11, fontweight='bold')
        if choice == 2 or choice == 3:
            ax0.plot(l,l*(l+1)*master_cl[l-2]*1e12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'g')
            ax0.plot(l,l*(l+1)*masked_cl[l-2]*1e12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'y')
            ax1.plot(l,l*(l+1)*delta_master[l-2]/(2*np.pi),'g')
        else:
            ax0.plot(l,l*(l+1)*master_cl[l]*1e12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'g')
            ax0.plot(l,l*(l+1)*masked_cl[l]*1e12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'y')
            ax1.plot(l,l*(l+1)*delta_master[l]/(2*np.pi),'g')
        ax0.plot(l,l*(l+1)*full_sky_cl[l]*1e12/bl[l]/bl[l]/pl[l]/pl[l]/(2*np.pi),'r')
        ax0.plot(l,l*(l+1)*cl[l]/(2*np.pi),'b')
        ax0.legend(["MASTER $C_l$ ","Masked $C_l$","full sky $C_l$","theoretical $C_l$" ], loc ="upper right")
        ax1.plot(l,l*(l+1)*delta_fs[l]/(2*np.pi),'r')

        ax1.yaxis.set_label_position("left")
        ax0.yaxis.set_label_position("left")
        plt.savefig("Plot_"+str(write_time)+"_"+str(LMAX)+".pdf", format='pdf')
        
        
def copy(cl,window_cl,full_sky_cl,masked_cl,master_cl,LMAX,c_t_i,NSIDE):
    os.system("mkdir fits_"+str(write_time)+"_files")
    if c_t_i == 1:
        full_sky_map =hp.synfast(cl,NSIDE,lmax=LMAX,alm=False,pol=False,pixwin=False,fwhm=0.0,new=True)
        hp.write_map("full_sky_unmasked_"+str(write_time)+"_.fits",full_sky_map)
        os.system("mv full_sky_unmasked_"+str(write_time)+"_.fits fits_"+str(write_time)+"_files")
    hp.write_cl("Theoretical_APS.fits", cl,overwrite = True)
    hp.write_cl("Window_APS.fits", window_cl,overwrite = True)
    hp.write_cl("Full_sky_APS.fits", full_sky_cl,overwrite= True)
    hp.write_cl("Masked_map_APS.fits", masked_cl,overwrite = True)
    hp.write_cl("Master_APS.fits", master_cl,overwrite = True)
    os.system("mv Theoretical_APS.fits Window_APS.fits Full_sky_APS.fits Masked_map_APS.fits Master_APS.fits Plot_"+str(write_time)+"_"+str(LMAX)+".pdf fits_"+str(write_time)+"_files")

    
    

def M_inv_l1_l2_EE_BB(window_cl,LMAX):
    M1 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
    N1 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

    M2 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
    N2 = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

    start_time = time.time()
    for l1 in range(2,LMAX+1):
        for l2 in range(l1,LMAX+1):
            N1[l1][l2] = N_l1_l2(l1,l2,LMAX,window_cl)[0]
            N2[l1][l2] = -N_l1_l2(l1,l2,LMAX,window_cl)[1]
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
    return M_inv
    
    
def M_inv_o(window_cl,LMAX):
    M = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
    N = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]

    s=0
    start_time = time.time()
    for l1 in range(2,LMAX+1):
        for l2 in range(l1,LMAX+1):
            N[l1][l2] = N_l1_l2(l1,l2,LMAX,window_cl)
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
    return M_inv

    
def temp_sum(l1,masked_cl,M_inv_i,LMAX):
    sp = 0
    for l2 in range(2,LMAX+1):
        sp = sp + M_inv_i[l1][l2]*masked_cl[l2]
    return sp

def get_master(masked_cl,LMAX,M_inv_i):
    l1  = 2
    mast_cl = np.zeros(LMAX+1)
    for i in range(2,LMAX+1):
        mast_cl[i] = temp_sum(l1,masked_cl,M_inv_i,LMAX)
        l1+=1
    return mast_cl  


def temp_sum_ee_bb(l1,masked_cl,M_inv_i,LMAX):
    s = 0
    for l2 in range(0,2*(LMAX-1)):
        s = s + M_inv_i[l1][l2]*masked_cl[l2]
    return s

def get_master_ee_bb(masked_cl,LMAX,M_inv_i):
    l1  = 0
    master_cl = np.zeros(2*(LMAX-1))
    for i in range(0,2*(LMAX-1)):
        master_cl[i] = temp_sum_ee_bb(l1,masked_cl,M_inv_i,LMAX)
        l1+=1
    return master_cl
    
def theory(c_t_i):
    #####Taking inputs#####
    cl_file = input("Enter the theory C_l file:")
    mask_file_t = input("Enter the name of the temperature mask file:")
    mask_file_p = input("Enter the name of the polarization mask file:")
    warnings.filterwarnings('ignore')
    mask_t = hp.read_map(mask_file_t)
    mask_p = hp.read_map(mask_file_p)
    NSIDE = hp.get_nside(mask_t)
    if hp.get_nside(mask_p) != NSIDE:
        print("The NSIDE of temperature and polarization masks are different!!")
        exit()
    LMAX=int(input("Enter LMAX:"))
    no =int(input("Enter no.of.realisations:"))
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
    
        
    window_cl_t = hp.anafast(mask_t ,lmax=LMAX)
    window_cl_p = hp.anafast(mask_p ,lmax=LMAX)
    window_cl_tp= hp.anafast(mask_t, mask_p ,lmax=LMAX)
    
    
    if choice == 2 or choice == 3:
        M_inv = [ [ 1 for i in range(2*(LMAX-1)) ] for j in range(2*(LMAX-1)) ]
        M_inv = M_inv_l1_l2_EE_BB(window_cl_p,LMAX)
    else:
        M_inv = [ [ 1 for i in range(LMAX+1) ] for j in range(LMAX+1) ]
        if choice == 1:
            M_inv = M_inv_o(window_cl_t,LMAX)
        elif choice == 4 or choice == 5:
            M_inv = M_inv_o(window_cl_tp,LMAX)
        elif choice == 6:
            M_inv = M_inv_o(window_cl_p,LMAX)
        

    cl=hp.read_cl(cl_file)
    clr=np.zeros(len(cl[0]))
    cl=np.vstack([cl,clr])
    cl=np.vstack([cl,clr])
    full_sky_cl_tt = np.zeros(LMAX+1)
    masked_cl_tt = np.zeros(LMAX+1)
    full_sky_cl_ee = np.zeros(LMAX+1)
    masked_cl_ee = np.zeros(LMAX+1)
    full_sky_cl_bb = np.zeros(LMAX+1)
    masked_cl_bb = np.zeros(LMAX+1)
    full_sky_cl_te = np.zeros(LMAX+1)
    masked_cl_te = np.zeros(LMAX+1)
    full_sky_cl_tb = np.zeros(LMAX+1)
    masked_cl_tb = np.zeros(LMAX+1)
    full_sky_cl_eb = np.zeros(LMAX+1)
    masked_cl_eb = np.zeros(LMAX+1)
    
    
    for i in range(no):
        full_sky_map =hp.synfast(cl,NSIDE,lmax=LMAX,alm=False,pol=False,pixwin=False,fwhm=0.0,new=True)
        masked_map = [ [ 1 for i in range(12*NSIDE**2) ] for j in range(3)]
        masked_map[0] = full_sky_map[0]*mask_t
        masked_map[1] = full_sky_map[1]*mask_p
        masked_map[2] = full_sky_map[2]*mask_p
        cl_sim = hp.anafast(full_sky_map, lmax=LMAX,pol=True)
        cl_mask_sim = hp.anafast(masked_map, lmax=LMAX,pol=True)
        full_sky_cl_tt = full_sky_cl_tt + cl_sim[0]/no
        masked_cl_tt = masked_cl_tt + cl_mask_sim[0]/no
        full_sky_cl_ee = full_sky_cl_ee + hp.anafast(full_sky_map[1], lmax=LMAX)/no
        full_sky_cl_bb =full_sky_cl_bb + hp.anafast(full_sky_map[2], lmax=LMAX)/no
        masked_cl_ee = masked_cl_ee + hp.anafast(masked_map[1],lmax=LMAX)/no
        masked_cl_bb = masked_cl_bb + hp.anafast(masked_map[2],lmax=LMAX)/no
        full_sky_cl_te =full_sky_cl_te + hp.anafast(full_sky_map[0],full_sky_map[1], lmax=LMAX)/no
        masked_cl_te = masked_cl_te + hp.anafast(masked_map[0],masked_map[1], lmax=LMAX)/no
        full_sky_cl_tb = full_sky_cl_tb + cl_sim[4]/no
        masked_cl_tb = masked_cl_tb + cl_mask_sim[4]/no
        full_sky_cl_eb = full_sky_cl_eb + cl_sim[5]/no
        masked_cl_eb = masked_cl_eb + cl_mask_sim[5]/no
        

    print("Time taken for the program to run is:",(time.time()-p_time)/60 ," minutes")
    if choice == 1:
        master_cl_tt = get_master(masked_cl_tt,LMAX,M_inv)
        PLOT_FUNC(cl[0],master_cl_tt,full_sky_cl_tt,masked_cl_tt,LMAX,no,NSIDE,c_t_i)
        copy(cl[0],window_cl_t,full_sky_cl_tt,masked_cl_tt,master_cl_tt,LMAX,c_t_i,NSIDE)
        os.system("mv fits_"+str(write_time)+"_files fits_TT_"+str(write_time)+"_files")
    elif choice == 4:
        master_cl_te = get_master(masked_cl_te,LMAX,M_inv)
        PLOT_FUNC(cl[3],master_cl_te,full_sky_cl_te,masked_cl_te,LMAX,no,NSIDE,c_t_i)
        copy(cl[3],window_cl_tp,full_sky_cl_te,masked_cl_te,master_cl_te,LMAX,c_t_i,NSIDE)
        os.system("mv fits_"+str(write_time)+"_files fits_TE_"+str(write_time)+"_files")
    elif choice == 5:
        master_cl_tb = get_master(masked_cl_tb,LMAX,M_inv)
        PLOT_FUNC(cl[4],master_cl_tb,full_sky_cl_tb,masked_cl_tb,LMAX,no,NSIDE,c_t_i)
        copy(cl[4],window_cl_tp,full_sky_cl_tb,masked_cl_tb,master_cl_tb,LMAX,c_t_i,NSIDE)
        os.system("mv fits_"+str(write_time)+"_files fits_TB_"+str(write_time)+"_files")
    elif choice == 6:
        master_cl_eb = get_master(masked_cl_eb,LMAX,M_inv)
        PLOT_FUNC(cl[5],master_cl_eb,full_sky_cl_eb,masked_cl_eb,LMAX,no,NSIDE,c_t_i)
        copy(cl[5],window_cl_p,full_sky_cl_eb,masked_cl_eb,master_cl_eb,LMAX,c_t_i,NSIDE)
        os.system("mv fits_"+str(write_time)+"_files fits_EB_"+str(write_time)+"_files")
        
    
    elif choice == 2 or choice == 3:
        masked_cl_ee = masked_cl_ee[2:]
        masked_cl_bb = masked_cl_bb[2:]

        mask_cl = np.concatenate([masked_cl_ee,masked_cl_bb])
        master_cl = get_master_ee_bb(mask_cl,LMAX,M_inv)


        master_cl_ee = master_cl[:len(master_cl)//2]
        master_cl_bb = master_cl[len(master_cl)//2:]
    
        if choice == 2:
            PLOT_FUNC(cl[1],master_cl_ee,full_sky_cl_ee,masked_cl_ee,LMAX,no,NSIDE,c_t_i)
            copy(cl[1],window_cl_p,full_sky_cl_ee,masked_cl_ee,master_cl_ee,LMAX,c_t_i,NSIDE)
            os.system("mv fits_"+str(write_time)+"_files fits_EE_"+str(write_time)+"_files")
        elif choice == 3:
            PLOT_FUNC(cl[2],master_cl_bb,full_sky_cl_bb,masked_cl_bb,LMAX,no,NSIDE,c_t_i)
            copy(cl[2],window_cl_p,full_sky_cl_bb,masked_cl_bb,master_cl_bb,LMAX,c_t_i,NSIDE)
            os.system("mv fits_"+str(write_time)+"_files fits_BB_"+str(write_time)+"_files")
            
            

            
def input_maps(c_t_i):
    print("Currently under development")
    
def ch_theory_input():
    c = int(input("Enter 1 if you want to generate Master spectrum using simulated one or enter 2 if you want to generate using input maps:"))


    if c ==1:
        print("You have selected Theory spectra!!!")
        theory(c)
    if c ==2:
        print("You have selected Input maps!!!")
        input_maps(c)
    print("Goodbye!!")

    
print("This code plots MASTER spectrum of theoretical simulations as well as PLANCK input maps.")
print("Choose one of the following options:\n")
print("1. TT spectrum\n2. EE spectrum\n3. BB spectrum\n4. TE spectrum\n5. TB spectrum\n6. EB spectrum\n")

choice = int(input("Enter desired option: "))

ch_theory_input()
    
