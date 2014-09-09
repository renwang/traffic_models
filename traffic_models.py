import matplotlib.pyplot as plt
from numpy import *
from random import choice
from copy import deepcopy


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


def ctm_qq(rho, model, Lambda, bdl, bdr, rhoc, rhom, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd):
        
## The quadratic quadratic(qq) model
## OUTPUT : traffic state at time n+1
## Description of parameters

#  rho : system state at time n
#  model: system model at time n, a vector which specifies the number of lanes open at each cell
#  Lambda =dt/dx    : discretization parameter in the Godunov scheme
#  bdl=[rhol,modell] : upstream density and model
#  bdr=[rhor,modelr] : downstream density and model
#  rhoc: critical density
#  rhom: maximum density
#  Vmax: maximum speed
#  rhom_tuide: critical density for the greenshield model, used to define the free flow part of the fundamental diagram

# Ren Wang
# UIUC, June 2014

## attach the boundary condition
        USrho = hstack((bdl[0],rho[0:-1]))
        DSrho = hstack((rho[1:],bdr[0]))
        USmodel = hstack((bdl[1],model[0:-1]))
        DSmodel = hstack((model[1:],bdr[1]))
## update density
        rho = rho + Lambda*(flux_qq(USrho, rho, rhoc, rhom, Vmax, USmodel,model,rhom_tuide) - flux_qq(rho, DSrho, rhoc, rhom, Vmax, model, DSmodel,rhom_tuide))+random.normal(ModelNoiseMean, ModelNoiseStd, (len(rho)))
        return rho
        

def vel_qq(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## the velocity function for the quadratic-quadratic model
        
# compute the parameters for the fundamental diagram        
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model
# initialize parameters used in this function
        cellNumber = len(rho)
        v1 = zeros((cellNumber))
        v2 = zeros((cellNumber))
        qmax = zeros((cellNumber))
        a = zeros((cellNumber))
        b = zeros((cellNumber))
        c = zeros((cellNumber))
## find the indexs where rhom and rho do not equal to zero
        Index_rhom = where(rhom != 0)
        Index_rho = where(rho !=0)
## compute qmax for the dimensions that rhoc and rhom are not zero
        qmax[Index_rhom] = rhoc[Index_rhom]*Vmax*(1-rhoc[Index_rhom]/rhom_tuide[Index_rhom])
## compute a b c for the cells where rhom does not equal to zero
        a[Index_rhom] = qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        b[Index_rhom] = -2*rhoc[Index_rhom]*qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        c[Index_rhom] = qmax[Index_rhom]*(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom])/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0).__and__(rhom != 0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        v0 = Vmax
        v1[Index_rhom] = Vmax*(1-rho[Index_rhom]/rhom_tuide[Index_rhom])
        v2[Index_rho] = (a[Index_rho]*rho[Index_rho]*rho[Index_rho]+b[Index_rho]*rho[Index_rho]+c[Index_rho])/rho[Index_rho]
        v3 = 0.0
        v = case0*v0+case1*v1+case2*v2+case3*v3
        return v

def vel_qq_scalar(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## the scalar velocity function for the quadratic-quadratic model
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model
        if rho<=0 and rhom!=0:
                v = Vmax
        elif rho<rhoc and rhom!=0:
                v = Vmax*(1-rho/rhom_tuide)
        elif rho>=rhoc and rho<rhom and rhom !=0:
                qmax = rhoc*Vmax*(1-rhoc/rhom_tuide)
                a = qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                b = -2*rhoc*qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                c = qmax*(2*rhoc*rhom-rhom*rhom)/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                v = (a*rho*rho+b*rho+c)/rho
        else:
                v = 0.0
        return v

def flow_qq(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## the flow-density function for the quadratic-quadratic model
        
# compute the parameters for the fundamental diagram        
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model
# initialize parameters used in this function
        cellNumber = len(rho)
        f1 = zeros((cellNumber))
        f2 = zeros((cellNumber))
        qmax = zeros((cellNumber))
        a = zeros((cellNumber))
        b = zeros((cellNumber))
        c = zeros((cellNumber))
## find the indexs where rhom and rho do not equal to zero
        Index_rhom = where(rhom != 0)
## compute qmax for the dimensions that rhoc and rhom are not zero
        qmax[Index_rhom] = rhoc[Index_rhom]*Vmax*(1-rhoc[Index_rhom]/rhom_tuide[Index_rhom])
## compute a b c for the cells where rhom does not equal to zero
        a[Index_rhom] = qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        b[Index_rhom] = -2*rhoc[Index_rhom]*qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        c[Index_rhom] = qmax[Index_rhom]*(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom])/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        f0 = 0.0
        f1[Index_rhom] = rho*Vmax*(1-rho[Index_rhom]/rhom_tuide[Index_rhom])
        f2 = a*rho*rho+b*rho+c
        f3 = 0.0
        f = case0*f0+case1*f1+case2*f2+case3*f3
        return f

def flow_qq_scalar(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## the scalar flow-density function for the quadratic-quadratic model

# compute the parameters for the fundamental diagram        
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model
        
        if rho<=0:
                f = 0.0
        elif rho<rhoc and rhom!=0:
                f = rho*Vmax*(1-rho/rhom_tuide)
        elif rho>=rhoc and rho<rhom and rhom !=0:
                qmax = rhoc*Vmax*(1-rhoc/rhom_tuide)
                a = qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                b = -2*rhoc*qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                c = qmax*(2*rhoc*rhom-rhom*rhom)/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                f = (a*rho*rho+b*rho+c)
        else:
                f = 0.0
        return f

def sending_qq(rho,rhoc,rhom,Vmax,model,rhom_tuide):
## sending function
        case1 = rho <= model*rhoc
        case2 = rho > model*rhoc
        q = flow_qq(rho,rhoc,rhom,Vmax,model,rhom_tuide)
        qmax = flow_qq(rhoc*model,rhoc,rhom,Vmax,model,rhom_tuide)
        qSend = case1*q+case2*qmax
        return qSend

def receiving_qq(rho,rhoc,rhom,Vmax,model,rhom_tuide):
## receiving function
        case1 = rho <= model*rhoc
        case2 = rho > model*rhoc
        qmax = flow_qq(rhoc*model,rhoc,rhom,Vmax,model,rhom_tuide)
        q = flow_qq(rho,rhoc,rhom,Vmax,model,rhom_tuide)
        qSend = case1*qmax+case2*q
        return qSend

def flux_qq(USrho,DSrho,rhoc,rhom,Vmax,USmodel,DSmodel,rhom_tuide):
## flux between cell boundary
        s_flux = sending_qq(USrho,rhoc,rhom,Vmax,USmodel,rhom_tuide)
        r_flux = receiving_qq(DSrho,rhoc,rhom,Vmax,DSmodel,rhom_tuide)
        flux = minimum(s_flux, r_flux)
        return flux


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

def ctm_qq_2nd(U, model, Lambda, bdl, bdr, rhoc1, rhoc2, rhom1, rhom2, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd):
# Description of model parameters:

# U = [rho; w] : traffic state at time n
# rho : traffic density state
# w     : represents the property of drivers/ fraction of cars
# model : a vector which specifies the number of lanes open at each cell
# Output           : traffic state at time n+1
# Lambda =dt/dx    : parameter in the cell transmission model/ the Godunov scheme
# bdl = [rhol, wl, modell] : upstream density and velocity boundary condition
# bdr = [rhor, wr, modelr] : downstream density and velocity boundary condtion
# rhom1 : upper bound of jam density (vehs)/lane
# rhom2 : lower bound of jam density (vehs)/lane
# rhoc1 : upper bound of critical density (vehs)/lane
# rhoc2 : lower bound of critical density (vehs)/lane
# Vmax  : the maximum velocity (speed limits)
#  rhom_tuide: critical density for the greenshield model, used to define the free flow part of the fundamental diagram


# Ren Wang
# UIUC, May 2014

# convert state = [rho; w] into conservation quantities [rho rho*w]Q = convertState(U)
        Q = convert_state(U)
        
# attach the boundary condtion
        USrho = hstack((bdl[0],U[0,0:-1]))
        DSrho = hstack((U[0,1:],bdr[0]))
        USw = hstack((bdl[1],U[1,0:-1]))
        DSw = hstack((U[1,1:],bdr[1]))
        USmodel = hstack((bdl[2],model[0:-1]))
        DSmodel = hstack((model[1:],bdr[2]))
# decompose state
        rho = U[0]
        w = U[1]
# update density
        Q = Q + Lambda*(flux_qq_2nd(USrho,rho, USw, w, rhoc1, rhoc2, rhom1, rhom2, Vmax,USmodel,model,rhom_tuide)-flux_qq_2nd(rho,DSrho, w, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax,model,DSmodel,rhom_tuide))
# augment the state
        rho = Q[0,:]+random.normal(ModelNoiseMean, ModelNoiseStd, (len(rho)))
        w = Q[1,:]/rho
        U = vstack((rho,w))
        return U


def convert_state(U):
        Q = vstack((U[0],U[0]*U[1]))
        return Q


def vel_qq_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## the velocity function for the quadratic-quadratic model
        
# compute the parameters for the fundamental diagram
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide

# initialize parameters used in this function
        cellNumber = len(rho)
        v1 = zeros((cellNumber))
        v2 = zeros((cellNumber))
        qmax = zeros((cellNumber))
        a = zeros((cellNumber))
        b = zeros((cellNumber))
        c = zeros((cellNumber))
## find the indexs where rhom and rho do not equal to zero
        Index_rhom = where(rhom != 0)
        Index_rho = where(rho !=0)
## compute qmax for the dimensions that rhoc and rhom are not zero
        qmax[Index_rhom] = rhoc[Index_rhom]*Vmax*(1-rhoc[Index_rhom]/rhom_tuide[Index_rhom])
## compute a b c for the cells where rhom does not equal to zero
        a[Index_rhom] = qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        b[Index_rhom] = -2*rhoc[Index_rhom]*qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        c[Index_rhom] = qmax[Index_rhom]*(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom])/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0).__and__(rhom != 0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        v0 = Vmax
        v1[Index_rhom] = Vmax*(1-rho[Index_rhom]/rhom_tuide[Index_rhom])
        v2[Index_rho] = (a[Index_rho]*rho[Index_rho]*rho[Index_rho]+b[Index_rho]*rho[Index_rho]+c[Index_rho])/rho[Index_rho]
        v3 = 0.0
        v = case0*v0+case1*v1+case2*v2+case3*v3
        return v

def vel_qq_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## the scalar velocity function for the quadratic-quadratic model
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide
        if rho<=0 and rhom!=0:
                v = Vmax
        elif rho<rhoc and rhom!=0:
                v = Vmax*(1-rho/rhom_tuide)
        elif rho>=rhoc and rho<rhom and rhom !=0:
                qmax = rhoc*Vmax*(1-rhoc/rhom_tuide)
                a = qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                b = -2*rhoc*qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                c = qmax*(2*rhoc*rhom-rhom*rhom)/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                v = (a*rho*rho+b*rho+c)/rho
        else:
                v = 0.0
        return v


def flow_qq_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## the flow-density function for the quadratic-quadratic model
        
# compute the parameters for the fundamental diagram        
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide
# initialize parameters used in this function
        cellNumber = len(rho)
        f1 = zeros((cellNumber))
        f2 = zeros((cellNumber))
        qmax = zeros((cellNumber))
        a = zeros((cellNumber))
        b = zeros((cellNumber))
        c = zeros((cellNumber))
## find the indexs where rhom and rho do not equal to zero
        Index_rhom = where(rhom != 0)
## compute qmax for the dimensions that rhoc and rhom are not zero
        qmax[Index_rhom] = rhoc[Index_rhom]*Vmax*(1-rhoc[Index_rhom]/rhom_tuide[Index_rhom])
## compute a b c for the cells where rhom does not equal to zero
        a[Index_rhom] = qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        b[Index_rhom] = -2*rhoc[Index_rhom]*qmax[Index_rhom]/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
        c[Index_rhom] = qmax[Index_rhom]*(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom])/(2*rhoc[Index_rhom]*rhom[Index_rhom]-rhom[Index_rhom]*rhom[Index_rhom]-rhoc[Index_rhom]*rhoc[Index_rhom])
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        f0 = 0.0
        f1[Index_rhom] = rho*Vmax*(1-rho[Index_rhom]/rhom_tuide[Index_rhom])
        f2 = a*rho*rho+b*rho+c
        f3 = 0.0
        f = case0*f0+case1*f1+case2*f2+case3*f3
        return f

def flow_qq_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## the scalar flow-density function for the quadratic-quadratic model

# compute the parameters for the fundamental diagram        
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide
        
        if rho<=0:
                f = 0.0
        elif rho<rhoc and rhom!=0:
                f = rho*Vmax*(1-rho/rhom_tuide)
        elif rho>=rhoc and rho<rhom and rhom !=0:
                qmax = rhoc*Vmax*(1-rhoc/rhom_tuide)
                a = qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                b = -2*rhoc*qmax/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                c = qmax*(2*rhoc*rhom-rhom*rhom)/(2*rhoc*rhom-rhom*rhom-rhoc*rhoc)
                f = (a*rho*rho+b*rho+c)
        else:
                f = 0.0
        return f

def sending_qq_2nd(rho,w,rhoc1,rhoc2,rhom1,rhom2,Vmax,model,rhom_tuide):
## sending function
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        
        case1 = rho <= rhoc
        case2 = rho > rhoc
        
        q = flow_qq_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide)
        qmax = flow_qq_2nd(rhoc, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide)
        qSend = case1*q+case2*qmax
        return qSend

def receiving_qq_2nd(rho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide):
## receiving function
        rhoc_USw = DSmodel*(rhoc1*rhoc2/(rhoc2*USw+rhoc1*(1-USw)))
        rhom_USw = DSmodel*(rhom1*rhom2/(rhom2*USw+rhom1*(1-USw)))
        
# compute the middle state        
        rho_middle = Riemman_qq(rho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)

# use the middle state to compute the receiving flow
        case1 = rho_middle <= rhoc_USw
        case2 = rho_middle > rhoc_USw

        qmax = flow_qq_2nd(rhoc_USw, USw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)
        q = flow_qq_2nd(rho_middle, USw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)
        
        qReceive = case1*qmax + case2*q
        return qReceive

def Riemman_qq(rho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide):
## solve the middle state in the riemman problem

# initialization        
        cellNumber=len(DSmodel)
        qmax = zeros((cellNumber))
        a = zeros((cellNumber))
        b = zeros((cellNumber))
        c = zeros((cellNumber))
        rho_middle = zeros((cellNumber))
        rho_middle1 = zeros((cellNumber))
        rho_middle2 = zeros((cellNumber))

# compute downstream cell velocity and the critical velocity        
        DSvel = vel_qq_2nd(rho, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)

# compute model parameters
        rhoc = DSmodel*(rhoc1*rhoc2/(rhoc2*USw+rhoc1*(1-USw)))
        rhom = DSmodel*(rhom1*rhom2/(rhom2*USw+rhom1*(1-USw)))
        Index = where(rhom != 0)

# compute the critical speed
        vc = vel_qq_2nd(rhoc, USw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)
        
        rhom_tuide = DSmodel*rhom_tuide
        
        qmax[Index] = rhoc[Index]*Vmax*(1-rhoc[Index]/rhom_tuide[Index])

        a[Index] = qmax[Index]/(2*rhoc[Index]*rhom[Index]-rhom[Index]*rhom[Index]-rhoc[Index]*rhoc[Index])
        b[Index] = -2*rhoc[Index]*qmax[Index]/(2*rhoc[Index]*rhom[Index]-rhom[Index]*rhom[Index]-rhoc[Index]*rhoc[Index])
        c[Index] = qmax[Index]*(2*rhoc[Index]*rhom[Index]-rhom[Index]*rhom[Index])/(2*rhoc[Index]*rhom[Index]-rhom[Index]*rhom[Index]-rhoc[Index]*rhoc[Index])

# compute middle density
        case1 = DSvel > vc
        case2 = DSvel <= vc

        rho_middle1[Index] = rhom_tuide[Index]*(1-DSvel[Index]/Vmax)
        rho_middle2[Index] = ((DSvel[Index]-b[Index])-((b[Index]-DSvel[Index])**2-4*a[Index]*c[Index])**0.5)/(2*a[Index])

        rho_middle[Index] = case1[Index]*rho_middle1[Index] + case2[Index]*rho_middle2[Index]
        return rho_middle

def flux_qq_2nd(USrho,DSrho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax,USmodel,DSmodel,rhom_tuide):
## flux between cell boundary
        s_flux = sending_qq_2nd(USrho,USw,rhoc1,rhoc2,rhom1,rhom2,Vmax,USmodel,rhom_tuide)
        r_flux = receiving_qq_2nd(DSrho,USw,DSw,rhoc1,rhoc2,rhom1,rhom2,Vmax,DSmodel,rhom_tuide)
        flux = minimum(s_flux, r_flux)
        flux_w = USw*flux
        Flux = vstack((flux, flux_w))
        return Flux

def qq_func_diff_v_w(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model,rhom_tuide):
## this is a function used to conduct Newton iteration for finding the boundary condition w in the second order model
        eps = 1e-2
        dv = vel_qq_scalar_2nd(rho, w+eps, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide) - vel_qq_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide) 
#        print 'here', vel_qq_scalar_2nd(rho, w+eps, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide), vel_qq_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide)
        y = dv/eps
        return y


def boundary_qq(bdl,bdr,rhoc1,rhoc2,rhom1,rhom2,Vmax,rhom_tuide, NewtonIteration):
## compute the boundary condition of the property parameter w for the second order model
        rhol = bdl[0]
        vell = bdl[1]
        modell = bdl[2]

        rhor = bdr[0]
        velr = bdr[1]
        modelr = bdr[2]

        wl = 0.5
        wr = 0.5

        if rhol<rhoc2*modell:
                pass
        else:
##                print vel_qq_scalar(rhol, rhoc2, rhom2, Vmax, modell, rhom_tuide)
##                print vel_qq_scalar(rhol, rhoc1, rhom1, Vmax, modell, rhom_tuide)
##                print vell
                if vell<=vel_qq_scalar(rhol, rhoc2, rhom2, Vmax, modell, rhom_tuide):
                        wl = 0
                elif vell>vel_qq_scalar(rhol, rhoc1, rhom1, Vmax, modell, rhom_tuide):
                        wl = 1
                else:
                        for i in range(NewtonIteration):
##                                print rhol,wl
                                diff_wl = qq_func_diff_v_w(rhol, wl, rhoc1, rhoc2, rhom1, rhom2, Vmax, modell, rhom_tuide)
                                if diff_wl == 0:
#                                        print 'diff_wl is 0'
                                        break
                                else:
                                        wl = wl - (vel_qq_scalar_2nd(rhol, wl, rhoc1, rhoc2, rhom1, rhom2, Vmax, modell, rhom_tuide)-vell)/diff_wl
                                        print 'wl is',wl                                
##                                        print 'vell is',vell
#                                        print 'current vell is',(vell-vel_qq_scalar_2nd(rhol, wl, rhoc1, rhoc2, rhom1, rhom2, Vmax, modell, rhom_tuide))
#                                        print 'diff_wl is', diff_wl
                                        
                        wl = min(max(wl,0), 1)

        if rhor<rhoc2*modelr:
                pass
        else:
                if velr<=vel_qq_scalar(rhor, rhoc2, rhom2, Vmax, modelr, rhom_tuide): 
                        wr = 0
                elif velr>vel_qq_scalar(rhor, rhoc1, rhom1, Vmax, modelr, rhom_tuide):
                        wr = 1
                else:
                        for i in range(NewtonIteration):
                                diff_wr = qq_func_diff_v_w(rhor, wr, rhoc1, rhoc2, rhom1, rhom2, Vmax, modelr, rhom_tuide)
                                if diff_wr == 0:
#                                        print 'diff_wr is 0'
                                        break
                                else:
                                        wr = wr - (vel_qq_scalar_2nd(rhor, wr, rhoc1, rhoc2, rhom1, rhom2, Vmax, modelr, rhom_tuide)-velr)/diff_wr
##                                        print 'wr is',wr                                
##                                        print 'velr is',velr
##                                        print 'current velr is',vel_qq_scalar(rhor, rhocr, rhomr, Vmax, modelr*rhom_tuide)                                        
                        wr = min(max(wr,0), 1)
        return (wl,wr)




########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################




def ctm_qt(rho, model, Lambda, bdl, bdr, rhoc, rhom, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd):

## The quadratic triangular(QT) model
## OUTPUT : traffic state at time n+1
## Description of parameters

#  rho : system state at time n
#  model: system model at time n, a vector which specifies the number of lanes open at each cell
#  Lambda =dt/dx    : discretization parameter in the Godunov scheme
#  bdl=[rhol,modell] : upstream density and model
#  bdr=[rhor,modelr] : downstream density and model
#  rhoc: critical density
#  rhom: maximum density
#  Vmax: maximum speed
#  rhom_tuide: critical density for the greenshield model, used to define the free flow part of the fundamental diagram

# Ren Wang
# UIUC, June 2014


## attach the boundary condition
        USrho = hstack((bdl[0],rho[0:-1]))
        DSrho = hstack((rho[1:],bdr[0]))
        USmodel = hstack((bdl[1],model[0:-1]))
        DSmodel = hstack((model[1:],bdr[1]))
## update density
        rho = rho + Lambda*(flux_qt(USrho, rho, rhoc, rhom, Vmax, USmodel,model,rhom_tuide) - flux_qt(rho, DSrho, rhoc, rhom, Vmax, model, DSmodel,rhom_tuide))+random.normal(ModelNoiseMean, ModelNoiseStd, (len(rho)))
        return rho


def vel_qt(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## quadratic triangular model velocity function for vector

# compute the parameters for the fundamental diagram        
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model
# initialize parameters used in this function
        cellNumber = len(rho)
        v1=zeros((cellNumber))
        v2=zeros((cellNumber))
        vc=zeros((cellNumber))
# find the indexs where rhom and rho do not equal to zero
        Index = where(minimum(rhom,rho) != 0)
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0).__and__(rhom != 0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        v0 = Vmax
        v1[Index] = Vmax*(1-(rho[Index]/rhom_tuide[Index]))
        vc[Index] = Vmax*(1-(rhoc[Index]/rhom_tuide[Index]))
        v2[Index] = rhoc[Index]*vc[Index]*(rhom[Index]-rho[Index])/(rhom[Index]-rhoc[Index])/rho[Index]
        v3 = 0.0
        v = case0*v0 + case1*v1 + case2*v2 + case3*v3
        return v

def vel_qt_scalar(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## quadratic triangular model velocity function for scalar
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model

        if rho<0 and rhom != 0:
                v = Vmax
        elif rho<=rhoc and rhom != 0:
                v = Vmax*(1-(rho/rhom_tuide))
        elif rho>=rhoc and rho<rhom and rhom !=0:
                vc = Vmax*(1-(rhoc/rhom_tuide))
                v = rhoc*vc*(rhom-rho)/(rhom-rhoc)/rho
        else:
                v = 0.0
        return v

def flow_qt(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## quadratic triangular model fundamental diagram for vector

# compute the parameters for the fundamental diagram        
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model
# initialize parameters used in this function
        cellNumber = len(rho)
        f1 = zeros((cellNumber))
        f2 = zeros((cellNumber))
        vc = zeros((cellNumber))
# find the indexs where rhom and rho do not equal to zero
        Index = where(rhom != 0)
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0).__and__(rhom != 0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        f0 = 0.0
        f1[Index] = rho[Index]*Vmax*(1-(rho[Index]/rhom_tuide[Index]))
        vc[Index] = Vmax*(1-(rhoc[Index]/rhom_tuide[Index]))
        f2[Index] = rhoc[Index]*vc[Index]*(rhom[Index]-rho[Index])/(rhom[Index]-rhoc[Index])
        f3 = 0.0
        f = case0*f0 + case1*f1 + case2*f2 + case3*f3
        return f

def flow_qt_scalar(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## quadratic triangular model velocity function for scalar

# compute the parameters for the fundamental diagram
        rhoc = rhoc*model
        rhom = rhom*model
        rhom_tuide = rhom_tuide*model

        if 0<rho<=rhoc and rhom != 0:
                f = rho*Vmax*(1-(rho/rhom_tuide))
        elif rho<rhom and rhom != 0:
                vc = Vmax*(1-(rhoc/rhom_tuide))
                f = rhoc*vc*(rhom-rho)/(rhom-rhoc)
        else:
                f = 0.0
        return f

def sending_qt(rho,rhoc,rhom,Vmax,model,rhom_tuide):
## sending function
        case1 = rho <= model*rhoc
        case2 = rho > model*rhoc
        q = flow_qt(rho,rhoc,rhom,Vmax,model,rhom_tuide)
        qmax = flow_qt(rhoc*model,rhoc,rhom,Vmax,model,rhom_tuide)
        qSend = case1*q+case2*qmax
        return qSend

def receiving_qt(rho,rhoc,rhom,Vmax,model,rhom_tuide):
## receiving function
        case1 = rho <= model*rhoc
        case2 = rho > model*rhoc
        qmax = flow_qt(rhoc*model,rhoc,rhom,Vmax,model,rhom_tuide)
        q = flow_qt(rho,rhoc,rhom,Vmax,model,rhom_tuide)
        qSend = case1*qmax+case2*q
        return qSend

def flux_qt(USrho,DSrho,rhoc,rhom,Vmax,USmodel,DSmodel,rhom_tuide):
## flux between cell boundary
        s_flux = sending_qt(USrho,rhoc,rhom,Vmax,USmodel,rhom_tuide)
        r_flux = receiving_qt(DSrho,rhoc,rhom,Vmax,DSmodel,rhom_tuide)
        flux = minimum(s_flux, r_flux)
        return flux



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

def ctm_qt_2nd(U,model, Lambda, bdl, bdr, rhoc1, rhoc2, rhom1, rhom2, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd):
# Description of model parameters:

# U = [rho; w] : traffic state at time n
# rho : traffic density state
# w     : represents the property of drivers/ fraction of cars
# model : a vector which specifies the number of lanes open at each cell
# Output           : traffic state at time n+1
# Lambda =dt/dx    : parameter in the cell transmission model/ the Godunov scheme
# bdl = [rhol, wl, modell] : upstream density and velocity boundary condition
# bdr = [rhor, wr, modelr] : downstream density and velocity boundary condtion
# rhom1 : upper bound of jam density (vehs)/lane
# rhom2 : lower bound of jam density (vehs)/lane
# rhoc1 : upper bound of critical density (vehs)/lane
# rhoc2 : lower bound of critical density (vehs)/lane
# Vmax  : the maximum velocity (speed limits)


# Ren Wang
# UIUC, May 2014

# convert state = [rho; w] into conservation quantities [rho rho*w]Q = convertState(U)
        Q = convert_state(U)
# attach the boundary condtion
        USrho = hstack((bdl[0],U[0,0:-1]))
        DSrho = hstack((U[0,1:],bdr[0]))
        USw = hstack((bdl[1],U[1,0:-1]))
        DSw = hstack((U[1,1:],bdr[1]))
        USmodel = hstack((bdl[2],model[0:-1]))
        DSmodel = hstack((model[1:],bdr[2]))
# decompose state
        rho = U[0]
        w = U[1]
# update density
        Q = Q + Lambda*(flux_qt_2nd(USrho,rho, USw, w, rhoc1, rhoc2, rhom1, rhom2, Vmax,USmodel,model,rhom_tuide)-flux_qt_2nd(rho,DSrho, w, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax,model,DSmodel,rhom_tuide))
# augment the state
        rho = Q[0,:]+random.normal(ModelNoiseMean, ModelNoiseStd, (len(rho)))
        w = Q[1,:]/rho
        U = vstack((rho,w))
        return U


def vel_qt_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## quadratic triangular model velocity function for vector

# compute the parameters for the fundamental diagram
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide

# initialize parameters used in this function
        cellNumber = len(rho)
        v1=zeros((cellNumber))
        v2=zeros((cellNumber))
        vc=zeros((cellNumber))
# find the indexs where rhom and rho do not equal to zero
        Index = where(minimum(rhom,rho) != 0)
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0).__and__(rhom != 0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        v0 = Vmax
        v1[Index] = Vmax*(1-(rho[Index]/rhom_tuide[Index]))
        vc[Index] = Vmax*(1-(rhoc[Index]/rhom_tuide[Index]))
        v2[Index] = rhoc[Index]*vc[Index]*(rhom[Index]-rho[Index])/(rhom[Index]-rhoc[Index])/rho[Index]
        v3 = 0.0
        v = case0*v0 + case1*v1 + case2*v2 + case3*v3
        return v

def vel_qt_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## quadratic triangular model velocity function for scalar
        
# compute the parameters for the fundamental diagram
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide

        if rho<0 and rhom != 0:
                v = Vmax
        elif rho<=rhoc and rhom != 0:
                v = Vmax*(1-(rho/rhom_tuide))
        elif rho>=rhoc and rho<rhom and rhom !=0:
                vc = Vmax*(1-(rhoc/rhom_tuide))
                v = rhoc*vc*(rhom-rho)/(rhom-rhoc)/rho
        else:
                v = 0.0
        return v

def flow_qt_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide):
## quadratic triangular model fundamental diagram for vector

# compute the parameters for the fundamental diagram
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide
# initialize parameters used in this function
        cellNumber = len(rho)
        f1 = zeros((cellNumber))
        f2 = zeros((cellNumber))
        vc = zeros((cellNumber))
# find the indexs where rhom and rho do not equal to zero
        Index = where(rhom != 0)
## case0: special case; case1: free flow; case2: congested; case3: over congested
        case0 = (rho<0).__and__(rhom != 0)
        case1 = (rho>=0).__and__(rho < rhoc)
        case2 = (rho >=rhoc).__and__(rho <rhom)
        case3 = (rho>=rhom)
        f0 = 0.0
        f1[Index] = rho[Index]*Vmax*(1-(rho[Index]/rhom_tuide[Index]))
        vc[Index] = Vmax*(1-(rhoc[Index]/rhom_tuide[Index]))
        f2[Index] = rhoc[Index]*vc[Index]*(rhom[Index]-rho[Index])/(rhom[Index]-rhoc[Index])
        f3 = 0.0
        f = case0*f0 + case1*f1 + case2*f2 + case3*f3
        return f

def flow_qt_scalar_2nd(rho, rhoc, rhom, Vmax, model, rhom_tuide):
## quadratic triangular model velocity function for scalar

# compute the parameters for the fundamental diagram
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        rhom_tuide = model*rhom_tuide

        if 0<rho<=rhoc and rhom != 0:
                f = rho*Vmax*(1-(rho/rhom_tuide))
        elif rho<rhom and rhom != 0:
                vc = Vmax*(1-(rhoc/rhom_tuide))
                f = rhoc*vc*(rhom-rho)/(rhom-rhoc)
        else:
                f = 0.0
        return f

def sending_qt_2nd(rho,w,rhoc1,rhoc2,rhom1,rhom2,Vmax,model,rhom_tuide):
## sending function
        rhoc = model*(rhoc1*rhoc2/(rhoc2*w+rhoc1*(1-w)))
        rhom = model*(rhom1*rhom2/(rhom2*w+rhom1*(1-w)))
        
        case1 = rho <= rhoc
        case2 = rho > rhoc
        
        q = flow_qt_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide)
        qmax = flow_qt_2nd(rhoc, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide)
        qSend = case1*q+case2*qmax
        print 'q send is', qSend
        return qSend

def receiving_qt_2nd(rho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide):
## receiving function
        rhoc_USw = DSmodel*(rhoc1*rhoc2/(rhoc2*USw+rhoc1*(1-USw)))
        rhom_USw = DSmodel*(rhom1*rhom2/(rhom2*USw+rhom1*(1-USw)))
        
# compute the middle state        
        rho_middle = Riemman_qt(rho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)

# use the middle state to compute the receiving flow
        case1 = rho_middle <= rhoc_USw
        case2 = rho_middle > rhoc_USw

        qmax = flow_qt_2nd(rhoc_USw, USw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)
        q = flow_qt_2nd(rho_middle, USw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)
        
        qReceive = case1*qmax + case2*q
        return qReceive

def Riemman_qt(rho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide):
## solve the middle state in the riemman problem

# initialization        
        cellNumber = len(DSmodel)
        qmax = zeros((cellNumber))
        rho_middle = zeros((cellNumber))
        rho_middle1 = zeros((cellNumber))
        rho_middle2 = zeros((cellNumber))

# compute downstream cell velocity        
        DSvel = vel_qq_2nd(rho, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)

# compute model parameters
        rhoc = DSmodel*(rhoc1*rhoc2/(rhoc2*USw+rhoc1*(1-USw)))
        rhom = DSmodel*(rhom1*rhom2/(rhom2*USw+rhom1*(1-USw)))
        Index = where(rhom != 0)

# compute the critical speed
        vc = vel_qq_2nd(rhoc, USw, rhoc1, rhoc2, rhom1, rhom2, Vmax, DSmodel, rhom_tuide)
        
        rhom_tuide = DSmodel*rhom_tuide
        
        qmax[Index] = rhoc[Index]*Vmax*(1-rhoc[Index]/rhom_tuide[Index])

# compute middle density
        case1 = DSvel > vc
        case2 = DSvel <= vc

        rho_middle1[Index] = rhom_tuide[Index]*(1-DSvel[Index]/Vmax)
        rho_middle2[Index] = qmax[Index]*rhom[Index]/(qmax[Index]-DSvel[Index]*(rhoc[Index]-rhom[Index]))


        rho_middle[Index] = case1[Index]*rho_middle1[Index] + case2[Index]*rho_middle2[Index]
        return rho_middle

def flux_qt_2nd(USrho,DSrho, USw, DSw, rhoc1, rhoc2, rhom1, rhom2, Vmax,USmodel,DSmodel,rhom_tuide):
## flux between cell boundary
        s_flux = sending_qt_2nd(USrho,USw,rhoc1,rhoc2,rhom1,rhom2,Vmax,USmodel,rhom_tuide)
        r_flux = receiving_qt_2nd(DSrho,USw,DSw,rhoc1,rhoc2,rhom1,rhom2,Vmax,DSmodel,rhom_tuide)
        flux = minimum(s_flux, r_flux)
        print 'flux is',flux
        flux_w = USw*flux
        Flux = vstack((flux, flux_w))
        return Flux


def qt_func_diff_v_w(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model,rhom_tuide):
## this is a function used to conduct Newton iteration for finding the boundary condition w in the second order model
        eps = 1e-2
        dv = vel_qt_scalar_2nd(rho, w+eps, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide) - vel_qt_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide) 
#        print 'here', vel_qq_scalar_2nd(rho, w+eps, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide), vel_qq_scalar_2nd(rho, w, rhoc1, rhoc2, rhom1, rhom2, Vmax, model, rhom_tuide)
        y = dv/eps
        return y


def boundary_qt(bdl,bdr,rhoc1,rhoc2,rhom1,rhom2,Vmax,rhom_tuide, NewtonIteration):
## compute the boundary condition of the property parameter w for the second order model
        rhol = bdl[0]
        vell = bdl[1]
        modell = bdl[2]

        rhor = bdr[0]
        velr = bdr[1]
        modelr = bdr[2]

        wl = 0.5
        wr = 0.5

        if rhol<rhoc2*modell:
                pass
        else:
                if vell<=vel_qt_scalar(rhol, rhoc2, rhom2, Vmax, modell, rhom_tuide):
                        wl = 0
                elif vell>vel_qt_scalar(rhol, rhoc1, rhom1, Vmax, modell, rhom_tuide):
                        wl = 1
                else:
                        for i in range(NewtonIteration):
                                diff_wl = qt_func_diff_v_w(rhol, wl, rhoc1, rhoc2, rhom1, rhom2, Vmax, modell, rhom_tuide)
                                if diff_wl == 0:
                                        break
                                else:
                                        wl = wl - (vel_qt_scalar_2nd(rhol, wl, rhoc1, rhoc2, rhom1, rhom2, Vmax, modell, rhom_tuide)-vell)/diff_wl
                                        print 'wl is',wl    
                        wl = min(max(wl,0), 1)

        if rhor<rhoc2*modelr:
                pass
        else:
                if velr<=vel_qt_scalar(rhor, rhoc2, rhom2, Vmax, modelr, rhom_tuide): 
                        wr = 0
                elif velr>vel_qt_scalar(rhor, rhoc1, rhom1, Vmax, modelr, rhom_tuide):
                        wr = 1
                else:
                        for i in range(NewtonIteration):
                                diff_wr = qt_func_diff_v_w(rhor, wr, rhoc1, rhoc2, rhom1, rhom2, Vmax, modelr, rhom_tuide)
                                if diff_wr == 0:
                                        break
                                else:
                                        wr = wr - (vel_qt_scalar_2nd(rhor, wr, rhoc1, rhoc2, rhom1, rhom2, Vmax, modelr, rhom_tuide)-velr)/diff_wr
                        wr = min(max(wr,0), 1)
        return (wl,wr)


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################






## self test section:

## scalar test
##rho=-5
##rhoc=37.0
##rhom=240.0
##Vmax=65.0
##rhom_tuide=30000

### vector test
##dt=20.0/3600
##dx=65*dt
##bdl=[10,3]
##bdr=[10,3]
##Vmax=65
##rho=array([-5,10,100])
##model=array([3,3,3])
##rhoc=37.0
##rhom=240.0
##rhom_tuide=30000.0
##Lambda=dt/dx
##ModelNoiseMean=0
##ModelNoiseStd=0.001
##
##print rho
##rho=ctm_qq(rho, model, Lambda, bdl, bdr, rhoc, rhom, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd)
##print rho

## 2nd test
##dt=20.0/3600
##dx=65*dt
##bdl=[10,0.5,3]
##bdr=[10,0.5,3]
##Vmax=65
##rho=array([0.0,300,100])
##w = array([0.5,0.5,0.5])
##model=array([3,3,3])
##rhoc1=42.0
##rhoc2=33.0
##
##rhom1=245.0
##rhom2=235.0
##
##rhom_tuide=30000
##Lambda=dt/dx
##ModelNoiseMean=0
##ModelNoiseStd=0.0001
##
##U=vstack((rho, w))
##
##
##print 'previous state is', U
##U=ctm_qt_2nd(U, model, Lambda, bdl, bdr, rhoc1, rhoc2, rhom1, rhom2, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd)
###rho=ctm_qq(rho, model, Lambda, bdl, bdr, rhoc, rhom, Vmax, rhom_tuide, ModelNoiseMean, ModelNoiseStd)
##print 'updated state is',U
####
####
####print _flux_qq(rho,rho,rhoc,rhom,Vmax,model,model,rhom_tuide)
####
####
###### note for simulation: Vmax=70.0, rhom_tuide=600.0, or Vmax=65.0, rhom_tuide=30000.0


#### boundary
##bdl=[130,59.25,3]
##bdr=[120,48,3]
##Vmax=65
##
##rhoc1=42.0
##rhoc2=33.0
##
##rhom1=245.0
##rhom2=235.0
##
##rhom_tuide=30000
##NewtonIteration=5
##wl, wr=boundary_qq(bdl,bdr,rhoc1,rhoc2,rhom1,rhom2,Vmax,rhom_tuide, NewtonIteration)
##
##
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
