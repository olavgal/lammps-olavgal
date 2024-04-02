/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Olav Galteland, olav.galteland@ntnu.no
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include "pair_lj_spline.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "citeme.h"

static const char cite_pair_lj_spline[] =
  "pair_style lj/spline:\n\n"
  "@article{hafskjold2019thermodynamic,\n"
  "title={Thermodynamic properties of the 3D Lennard-Jones/spline model},\n"
  "author={Hafskjold, Bjørn and Travis, Karl Patrick and Hass, Amanda Bailey and Hammer, Morten and Aasen, Ailo and Wilhelmsen, Øivind},\n"
  "journal={Molecular Physics},\n"
  "volume={117},\n"
  "number={23-24},\n"
  "pages={3754--3769},\n"
  "year={2019},\n"
  "publisher={Taylor \\& Francis}\n"
  "}\n\n";

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJSpline::PairLJSpline(LAMMPS *lmp) : Pair(lmp){
  if (lmp->citeme)
    lmp->citeme->add(cite_pair_lj_spline);
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairLJSpline::~PairLJSpline(){
  if (allocated) {
    memory->destroy(setflag);

    memory->destroy(cut);
    memory->destroy(cutsq);
    memory->destroy(cut_inner);
    memory->destroy(cut_inner_sq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(radius);
    memory->destroy(radius_sq);
    memory->destroy(alpha);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(a);
    memory->destroy(b);
    memory->destroy(wall);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJSpline::compute(int eflag, int vflag){
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  double R, rc;
  double r,t,tsq,fskin;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  forcelj = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        R = radius[itype][jtype];
        if (rsq < cut_inner_sq[itype][jtype]) {
          r = sqrt(rsq);
          forcelj = (1.0/r)*(lj1[itype][jtype]*pow(r-R, -13) 
          - lj2[itype][jtype]*pow(r-R, -7));
        } 

        else {
          r = sqrt(rsq);
          rc = cut[itype][jtype];
          forcelj = ((rc-r)/r)*(2.0*a[itype][jtype]+3.0*b[itype][jtype]*(r-rc));
        }
        
        fpair = factor_lj*forcelj;
        
        if (!wall[itype][itype]){
          f[i][0] += delx*fpair;
          f[i][1] += dely*fpair;
          f[i][2] += delz*fpair;
        }
        if ((!wall[jtype][jtype]) && (newton_pair || j < nlocal)) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < R*R)
            evdwl = double(DBL_MAX);
          else if (rsq < cut_inner_sq[itype][jtype]){
            r = sqrt(rsq);
            evdwl = (lj3[itype][jtype]*pow(r-R, -12.0) - lj4[itype][jtype]*pow(r-R, -6.0));
          }
          else{
            r = sqrt(rsq);
            rc = cut[itype][jtype];
            evdwl = pow(r-rc, 2.0)*(a[itype][jtype]+b[itype][jtype]*(r-rc));
          }
        }
        if (evflag) 
          ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJSpline::allocate(){
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut_inner,n+1,n+1,"pair:cut_inner");
  memory->create(cut_inner_sq,n+1,n+1,"pair:cut_inner_sq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(radius,n+1,n+1,"pair:radius");
  memory->create(radius_sq,n+1,n+1,"pair:radius_sq");
  memory->create(alpha,n+1,n+1,"pair:alpha");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(a,n+1,n+1,"pair:a");
  memory->create(b,n+1,n+1,"pair:b");
  memory->create(wall,n+1,n+1,"pair:wall");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJSpline::settings(int narg, char **arg){
  if (narg != 0) 
    error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJSpline::coeff(int narg, char **arg){
  if (narg != 6 and narg != 7) 
    error->all(FLERR,"Incorrect args for pair coefficients");
    
  if (!allocated) 
    allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one    = utils::numeric(FLERR,arg[2], false, lmp);
  double sigma_one      = utils::numeric(FLERR,arg[3], false, lmp);
  double alpha_one      = utils::numeric(FLERR,arg[4], false, lmp);
  double radius_one     = utils::numeric(FLERR,arg[5], false, lmp);
  bool wall_one;

  if (narg == 7){
    double tmp = utils::numeric(FLERR,arg[6], false, lmp);
    if (tmp == 0)
      wall_one = false;
    else if (tmp == 1)
      wall_one = true;
    else
      error->all(FLERR,"Incorrect wall args for lj/spline coefficients");
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j]     = epsilon_one;
      sigma[i][j]       = sigma_one;
      alpha[i][j]       = alpha_one;
      radius[i][j]      = radius_one;
      if (narg == 7)
        wall[i][j]      = wall_one;
      else
        wall[i][j]      = false;
      setflag[i][j]     = 1;
      count++;
    }
  }

  if (count == 0) 
    error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJSpline::init_one(int i, int j){
  
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                                 sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    radius[i][j] = mix_distance(radius[i][i],radius[j][j]);
    alpha[i][j] = mix_distance(alpha[i][i],alpha[j][j]); 
  }
  radius[j][i] = radius[i][j];
  sigma[j][i] = sigma[i][j];
  epsilon[j][i] = epsilon[i][j];
  alpha[j][i] = alpha[i][j];

  cut_inner[i][j]     = pow(26.0/(7.0*alpha[i][j]), 1.0/6.0)
                          *(sigma[i][j]-radius[i][j]) + radius[i][j];
  a[i][j]             = -(24192.0/3211.0)*epsilon[i][j]*pow(alpha[i][j], 2.0)*
                          pow(cut_inner[i][j]-radius[i][j], -2.0);
  b[i][j]             = -(387072.0/61009.0)*epsilon[i][j]*pow(alpha[i][j], 2.0)*
                          pow(cut_inner[i][j]-radius[i][j], -3.0);

  cut[i][j]           = (67.0/48.0)*cut_inner[i][j] - (19.0/48.0)*radius[i][j];
  
  cut_inner_sq[i][j]  = cut_inner[i][j]*cut_inner[i][j];
  radius_sq[i][j]     = radius[i][j]*radius[i][j];
      
  // Precalculate constants
  lj1[i][j]       = 48.0*epsilon[i][j]*pow(sigma[i][j]-radius[i][j], 12.0);
  lj2[i][j]       = 24.0*epsilon[i][j]*alpha[i][j]*pow(sigma[i][j]-radius[i][j], 6.0);
  lj3[i][j]       = 4.0*epsilon[i][j]*pow(sigma[i][j]-radius[i][j], 12.0);
  lj4[i][j]       = 4.0*epsilon[i][j]*alpha[i][j]*pow(sigma[i][j]-radius[i][j], 6.0);

  // Making matrices symmetric
  cut_inner[j][i]     = cut_inner[i][j];
  cut_inner_sq[j][i]  = cut_inner_sq[i][j];
  cut[j][i]           = cut[i][j];
  radius_sq[j][i]     = radius_sq[i][j];
  a[j][i]             = a[i][j];
  b[j][i]             = b[i][j];
  lj1[j][i]           = lj1[i][j];
  lj2[j][i]           = lj2[i][j];
  lj3[j][i]           = lj3[i][j];
  lj4[j][i]           = lj4[i][j];
 
  return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSpline::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++){
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&radius[i][j],sizeof(double),1,fp);
        fwrite(&alpha[i][j],sizeof(double),1,fp);
        fwrite(&cut_inner[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
        fwrite(&a[i][j],sizeof(double),1,fp);
        fwrite(&b[i][j],sizeof(double),1,fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSpline::read_restart(FILE *fp){
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++){
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
        MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&radius[i][j],sizeof(double),1,fp);
          fread(&alpha[i][j],sizeof(double),1,fp);
          fread(&cut_inner[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
          fread(&a[i][j],sizeof(double),1,fp);
          fread(&b[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&radius[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&alpha[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_inner[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b[i][j],1,MPI_DOUBLE,0,world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSpline::write_restart_settings(FILE *fp)
{
  fwrite(&cut_inner_global,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSpline::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_inner_global,sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_inner_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
 proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJSpline::write_data(FILE *fp){
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",
      i,epsilon[i][i],sigma[i][i],radius[i][i],alpha[i][i]);
}

/* ----------------------------------------------------------------------
 proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJSpline::write_data_all(FILE *fp){
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g\n",i,j,
        epsilon[i][j],sigma[i][j],radius[i][j],
        alpha[i][j],cut_inner[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJSpline::single(int i, int j, int itype, int jtype, double rsq,
                          double factor_coul, double factor_lj,
                          double &fforce){
  double r, R, rc, forcelj, philj;
  R = radius[itype][jtype];
  if (rsq < cut_inner_sq[itype][jtype]) {
    r = sqrt(rsq);
    forcelj = (1.0/r)*(lj1[itype][jtype]*pow(r-R, -13) - lj2[itype][jtype]*pow(r-R, -7));
    philj = (lj3[itype][jtype]*pow(r-R, -12.0)-lj4[itype][jtype]*pow(r-R, -6.0));
  } 
  else {
    r = sqrt(rsq);
    rc = cut[itype][jtype];
    forcelj = ((rc-r)/r)*(2.0*a[itype][jtype]+3.0*b[itype][jtype]*(r-rc));
    philj = pow(r-rc, 2.0)*(a[itype][jtype]+b[itype][jtype]*(r-rc));
  }

  fforce = factor_lj*forcelj;
  return factor_lj*philj;
}
