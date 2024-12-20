CX      slightly modified output to the control log file Apr08/2005
CX    This is a modification of VENUS to do atom+polyatomic molecule
CX                   reactive and universal nonreactive scattering
CX        modifyed by G. Lendvay October 1994
CX        Parallelized  May 1996
CX          ---------Unparallelized version---------
CX   First step completed -- integration routines separated from main
CX       to be tested -- May07, 1996  passed test
CX   Second step -- delete normal mode population counts etc.
CX       to be tested -- May07, 1996  passed test
CX     Reaction-specific output replaced by en. transfer specific output
cx  read NT,NS,NIP,NCROT in free format
CXP
CX      The modifications of the original VENUS code made by G. Lendvay 
CX      are marked by CXCCa at the beginning and by CXCCz at the end 
CX      or sometimes simply by CX.
CX      AIX-specific and more recent (1991-1994) modifications are marked
CX      by CXAIX.
CXP
CXP     Parallelization modifications are marked by CXP
CXP
CXP      All      WRITE(6  statements are replaced by 
CXP         CXW...WRITE(6    to avoid delay in production run
CXP        after the end of test of input
CXP      When remowing, replace the entire 'CXW...WR' by  '      WR'
CXP        There are IF( ) WRITE cases also
CXP        There are some 5 continuation lines for WRITE(6,  in which 
CXP         CXW... is  easy to remove by hand
CXP      Check directing to 449 CALL GWRITE is also commented out by CXW...
CX
CX    In order to truncate the size of the code, most of the sophisticated 
CX      potential subroutines  of the original VENUS (which amount up to 30 
CX      but almost never used) are removed in this version. The potential
CX      building blocks that are kept are: 
CX          harmonic stretch,
CX          Morse stretch,
CX          harmonic bend,
CX          Lennard-Jones potential.
CX      Note that the alpha bend is replaced in this code, and the numbers
CX      of atoms that originally were inputted for alpha bends determine
CX      now which three atoms have the triatomic potential added separately.
CX      Any reference to the other potential types of VENUS is disregarded.
CX
CXAIX   To conform with the AIX fortran random number generator syntax, 
CXAIX   srand starts the sequence with REAL seed number RANDS=float(KRAND)
CXAIX         and RAND(KRAND) is modified to RAND() throughout the code
CXAIX    for srand see CXAIX001
CXSGI   To run on a Silicon Graphics machine, replace all occurences of
CXSGI   RANF() by RAND(KRAND)
CXP     The AIX->CRAY random number transformation is effected by
CXP      the replacement of all occurrences of RAND() by RANF()
cx6SEC   random numbers are generated via 
cx6SEC   ZBQLU01
cx  www.homepages.ucl.ac.uk/~ucakarc/work/randgen.html 
C***********************************************************************
C                                                                      *
C                            VENUS                                     *
C                                                                      *
C          A GENERAL CHEMICAL DYNAMICS COMPUTER PROGRAM                *
C                                                                      *
C                             BY                                       *
C                                                                      *
C   W.L. HASE, R.J. DUCHOVIC, D.-H. LU, K.N. SWAMY, S.R. VANDE LINDE,  *
C                        AND R.J. WOLF                                 *
C                                                                      *
C                      NOVEMBER 8, 1988                                *
C                                                                      *
C***********************************************************************
      IMPLICIT REAL*8 (A-H,O-Z)
CXP This common block collects trajectory data to be analyzed later
CXpoly   The parameters collected for a reactive system
CXpoly      COMMON/parall/ SBP(1024),EVIB0P(1024)
CXpoly     *,ERTB0P(1024),EVIBP(1024),EROTBP(1024),EVIAP(1024)
CXpoly     *,EROTAP(1024),ERELSP(1024),ANG4(1024),NPATHP(1024)
CXpoly     *,AMAIP(1024),AMAFP(1024),AMBIP(1924),AMBFP(1024)
CXENTR      COMMON/parall/ EVIB0P(1024),ERTB0P(1024),EVIBP(1024)
CXENTR     *,EROTBP(1024),EVIA0P(1024),ERTA0P(1024),EVIAP(1024)
CXENTR     *,EROTAP(1024),SER1P(1024),ERELSP(1024),SBP(1024)
CXENTR     *,COLLTP(1024),LINNERP(1024)
CXENTR     *,AMAIP(1024),AMAFP(1024),AMBIP(1924),AMBFP(1024)
CXUP  COMMON/parall/ EVIB0P(1024),ERTB0P(1024),EVIBP(1024)
CXUP *,EROTBP(1024),EROTAP(1024),SER1P(1024),ERELSP(1024),ANG4P(1024)
CXUP *,SBP(1024),COLLTP(1024),AMBIP(1924),AMBFP(1024),LINNERP(1024)
CXP
      COMMON/PRLIST/T,V,H,TIME,NTZ,NC,KRANOD ,NT
      COMMON/PRFLAG/NFQP,NFR,NUMR,NFB,NUMB,NFA,NUMA,NFTAU,NUMTAU,
     *NFTET,NUMTET,NFDH,NUMDH
      COMMON/PARRAY/KR(25),JR(25),IB(25),IA(10),ITAU(15),ITET(6),IDH(25)
      COMMON/SELTB/QZ(150),RANOD,KRAND,NSELT,NSFLAG,NACT,NLINA,NLINB
      COMMON/QPDOT/Q(150),PDOT(150)
      COMMON/PQDOT/P(150),QDOT(150),W(50)
      COMMON/HFIT/PSCALA,PSCALB,VZERO
      COMMON/FORCES/NATOMS,I3N,NST,NM,NB,NA,NLJ,NTAU,NEXP,NGHOST,
     *NTET,NVRR,NVRT,NVTT,NANG
      COMMON/STRETB/RSZ(100),FS(100),N1J(100),N1K(100)
      COMMON/MORSEB/RMZ(100),B(100),D(100),N2J(100),N2K(100),
     *CM1(100),CM2(100),CM3(100),CM4(100)
      COMMON/BENDB/THETAZ(100),FBZ(100),CJ(100),CK(100),RJZ(100),
     *RKZ(100),FB(100),N3J(100),N3K(100),N3M(100)
      COMMON/ALPHAB/FA(20),N4J(20),N4K(20),N4M(20),N4N(20)
      COMMON/LENJB/ALJ(500),BLJ(500),CLJ(500),N5J(500),N5K(500),NREP(500
     *),MREP(500),LREP(500)
      COMMON/TAUB/VZTAU(15),N6I(15),N6J(15),N6K(15),N6L(15),N6M(15),N6N(
     *15)
      COMMON/EXPB/AEX(100),BEX(100),CEX(100),N7J(100),N7K(100),NPOW(100)
      COMMON/GHOSTB/GC1(20),GEX1(20),GEX2(20),N8I(20),N8J(20),N8K(20),
     *N8L(20),N8M(20),N8N(20)
      COMMON/TETRAB/ N9I(20),N9J(20),N9K(20),N9L(20),N9M(20),
     *               FT0(20,6),FT2(20,6),GT0(20,6),GT2(20,6),
     *               HT0(20,6),HT2(20,6),THT(20,6),R0(20,4),
     *               THT1(20,6),THT2(20,6),FD1(20,4),
     *               HD1(20,4),GN0(20,5),FT(20,6),GT(20,6),
     *               HT(20,6),FD(20,4),HD(20,4),DLTA(20,48),
     *               TETTST,SGN1,SGN2,SGN3,SGN4
      COMMON/CUBEB/ S3(4),DS3(4),CBIC(15,6),ANG1(20,6,4),GN4(20)
      COMMON/VRRB/FKRRZ(100),FKRR(100),CIJ(100),CKL(100),RIJ0(100),
     *            RKL0(100),N10I(100),N10J(100),N10K(100),N10L(100)
      COMMON/VRTB/FKRTZ(50),FKRT(50),CRT(50),R110(50),
     *            N11I(50),N11J(50),N11B(50),NRT(50)
      COMMON/VTTB/FKTTZ(100),FKTT(100),N12B(100),N12BB(100)
      COMMON/ANGLEB/FDH(25,4),GDH(25,4),NDH(25),N12I(25),N12J(25),
     *N12K(25),N12L(25)
CXCCC
CXCCC COMMON/TESTIN/VRELO,INTST
      COMMON/TESTIN/VRELO,INTST,RCMO
CXCCC
      COMMON/COORS/R(500),THETA(100),ALPHA(20),CTAU(15),GR(20,5),
     *TT(20,6),DANG(25)
      COMMON/CONSTN/C1,C2,C3,C4,C5,C6,C7,PI,HALFPI,TWOPI
      COMMON/FRAGB/WTA(10),WTB(10),LA(10,25),LB(10,25),QZA(10,75),
     *QZB(10,75),NATOMA(10),NATOMB(10)
      COMMON/TESTB/RMAX(10),RBAR(10),NTEST,NPATHS,NABJ(10),NABK(10),
CXpoly    *NPATH
     *NABL(10),NPATH
      COMMON/FINALB/EROTA,EROTB,EA(3),EB(3),AMA(4),AMB(4),OAM(4),
     *EREL,ERELSQ,BF,SDA,SDB,DELH(10),ANG(20),NFINAL
      COMMON/CHEMAC/WWA(150),CA(150,150),AI(3),ENMTA,RZA,FAHARM,BA,DA,
     *    AMPA(150),WWB(150),CB(150,150),BI(3),ENMTB,RZB,FBHARM,BB,DB,
     *    AMPB(150),SEREL,S,BMAX,TROTA,TROTB,ANQA(150),ANQB(150),
     *    NROTA,NROTB,NOB
      COMMON/WASTE/QQ(150),PP(150),WX,WY,WZ,LL(25),NAM
CXCCa COMMON/HARMB/NNA,JA,NNB,JB,NHARMA,NHARMB
cx  non-integer quantum numbers for diatomic B
      COMMON/HARMB/NNA,JA,VNB,RJB,NHARMA,NHARMB,TVIBA,TVIBB
CXCCz
      COMMON/RSTART/HINC,NPTS
      COMMON/RKUTTA/RAA1,RA1,RA2,RA3,RB1,RB2,RB3,RC1,RC2
      COMMON/LMODEB/ENON,NONI,NONJ,JFLAG
CXCCa 
      COMMON/LGT/ HINTB,ERTB0,SERL,ajqai,ajqbi,ajqaf,ajqbf,SB
     *,HINTA,ERTA0,lfault,linner
      COMMON/LGT1/ TOTIN,SER1                       
      CHARACTER*40 FNAM1,FNAM2,FNAM3,fn49
      CHARACTER*1 fn2(40),fn3(40)
      EQUIVALENCE (FNAM2,fn2(1))
      EQUIVALENCE (FNAM3,fn3(1))
      dimension aja(500),ajb(500),ifai(100)
CX    COMMON/VECTB/VI(4),OAMI(4),AMAI(4),AMBI(4)
CXAIX    variables for collision time accounting
      COMMON/FINALTIM/RTIME1,RTIME2,RTIME3,RTIME4,TMIN
     @,RELKINO,DELKINO,CLOSEST
CXAIXa this block transfers the last internuclear separation to 
CX     GFINAL, where it is used to calculate COLLTIME
      COMMON/COLLTI1/ RCMFIN
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      common/isomer/diff, olddiff, somm,mini1,mini2,mini3,
     &  iTini1,iTini2,iTini3,iTfin1,iTfin2,iTfin3,iTtot1, iTtot2,iTtot3

      common/collti/COLLTIME
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
c@ extra commons for isomerization counter

      CHARACTER  FNAME1*40,FNAME2*40,FNAME3*40,fne49*40
      CHARACTER fne2(40),fne3(40)
      EQUIVALENCE (FNAME2,fne2(1))
      EQUIVALENCE (FNAME3,fne3(1))

c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CXCCz 
C************
      common/reducedmb/ wtbsz(10),redmassb(10)
      common/evj1/ einit,efinal,hinit,hfinal
      common/potcm/ r6sec(3),energy6sec,dedr6sec(3)
C************
CXP    BQ( and BP( are used in the separated integration routines now
CXP   DIMENSION BQ(150),BP(150),TABLE(2100),ERAVA(500),ERAVB(500)
      DIMENSION ERAVA(500),ERAVB(500)
     *,TITLE(36),QCM(3),VCM(3)
      DIMENSION ETIM(2000),ESAV(2000),ESQ(2000),NEVIB(2000),
     *NEVIBU(2000),NEVIBL(2000),NEVIBN(2000)
CXP      DIMENSION EBM(50),ENM(50),EBSAV(2,2000),EBSQ(2,2000),
CXP     *ENSAV(30,2000),ENSQ(30,2000)
      DIMENSION NLMI(50),NLMJ(50),RLMZ(50)
CXAIX001
      REAL RANDS
CXAIXz	
C
    1 FORMAT(18I4)
    2 FORMAT(2I10,3F10.0)
    3 FORMAT(3X,3HNJ=,I4,5H  NK=,I4,5H  RZ=,F6.3,4H  F=,F9.3)
    4 FORMAT(3X,3HNJ=,I4,5H  NK=,I4,5H  RZ=,F6.3,4H  B=,F6.3,4H  D=,F8.3
     *)
    5 FORMAT(3I10,7F15.0)
    6 FORMAT(3X,3HNJ=,I4,5H  NK=,I4,5H  NM=,I4,9H  THETAZ=,F8.3,5H  FZ=,
     *F9.3,6H  RJZ=,F6.3,5H  CJ=,F6.3,6H  RKZ=,F6.3,5H  CK=,F6.3)
    7 FORMAT(6F20.0)
    8 FORMAT(7I10)
    9 FORMAT(I12,F10.0)
   10 FORMAT(3X,10F10.5)
   11 FORMAT(3X,3HNT=,I5,5H  NS=,I10,6H  NIP=,I6,8H  NCROT=,I6)
   12 FORMAT(3X,6HKRAND=,I12,7H  TIME=,F9.5,//)
   13 FORMAT(3X,30HACTIVATE WITH ORTHANT SAMPLING,/)
   14 FORMAT(1X,16HNUMBER OF ATOMS=,I4,/,' NUMBER OF HARMONIC STRETCH'
     *,'ES=',I4,/,' NUMBER OF MORSE STRETCHES=',I4,/,' NUMBER OF ',
     * 'HARMONIC BENDS=',I4,/,23H NUMBER OF TRIATOMICS =,I4,/,' NUMBE',
     * 'R OF LENNARD-JONES INTERACTIONS=',I4,/,' NUMBER OF TORSIONS='
     * ,I4,/,22H NUMBER OF REPULSIONS=,I4,/,17H NUMBER OF GHOST   ,
     * 6HPAIRS=,I4,/,31H NUMBER OF TETRAHEDRAL CENTERS=,I4,/,
     * 25H NUMBER OF R-R COUPLINGS=,I4,/,
     * 29H NUMBER OF R-THETA COUPLINGS=,I4,/,
     * 33H NUMBER OF THETA-THETA COUPLINGS=,I4,/,
     * 27H NUMBER OF DIHEDRAL ANGLES=,I4,//)
   15 FORMAT(/)
   16 FORMAT(3X,5HNFQP=,I2,/)
   17 FORMAT(3X,4HNFR=,I2,6X,5HNUMR=,I2)
   18 FORMAT(6X,7HJ-ATOM=,I4,4X,7HK-ATOM=,I4)
   19 FORMAT(3X,4HNFB=,I2,6X,5HNUMB=,I2)
   20 FORMAT(15X,37HINDICES OF THETA ANGLES TO BE PRINTED)
   21 FORMAT(6X,20I4)
   22 FORMAT(3X,4HNFA=,I2,6X,5HNUMA=,I2)
   23 FORMAT(15X,37HINDICES OF ALPHA ANGLES TO BE PRINTED)
   24 FORMAT(3X,6HNFTAU=,I2,4X,7HNUMTAU=,I2)
   25 FORMAT(15X,35HINDICES OF TAU ANGLES TO BE PRINTED)
   26 FORMAT(3X,54HINITIAL CONDITIONS ARE CHOSEN FOR ONE OR TWO REACTANT
     *S)
   27 FORMAT(3X,38HINITIAL CONDITIONS ARE READ IN: NSELT=,I2,//)
   28 FORMAT(3X,16HMASSES OF ATOMS:,I3,6H ATOMS)
   29 FORMAT(4I10,5F15.0)
   30 FORMAT(3X,3HNJ=,I4,5H  NK=,I4,5H  NM=,I4,5H  NN=,I4,4H  F=,F7.3)
   31 FORMAT(3X,3HNJ=,I4,5H  NK=,I4,6H  ALJ=,D12.5,6H  BLJ=,D12.5,
     *6H  CLJ=,D12.5,7H  NREP=,I4,7H  MREP=,I4,7H  LREP=,I4)
   32 FORMAT(3X,3HNQ=,I4,5H  NP=,I4,//)
   33 FORMAT(6I10,F10.0)
   34 FORMAT(3X,3HNI=,I4,5H  NJ=,I4,5H  NK=,I4,5H  NL=,I4,5H  NM=,
     *I4,5H  NN=,I4,5H  VZ=,F7.3)
   35 FORMAT(3X,49HACTIVATE WITH MICROCANONICAL NORMAL MODE SAMPLING,/)
   36 FORMAT(7X,F10.6,1X,F10.6,1X,F10.6,3X,F10.6,1X,F10.6,1X,F10.6,
     *3X,F10.6,1X,F10.6,1X,F10.6)
   37 FORMAT(3X,3HNJ=,I4,5H  NK=,I4,7H  NPOW=,I4,6H  AEX=,D12.5,
     *6H  BEX=,D12.5,6H  CEX=,D12.5)
   38 FORMAT(3X,6HVZERO=,F8.3,10H KCAL/MOLE)
   39 FORMAT(3X,20HPARAMETERS FOR PATH ,I1,1H:,/,5X,5HRMAX=,F6.2,2X,5HRB
     *AR=,F6.2,2X,7HNATOMA=,I3,2X,7HNATOMB=,I3,2X,5HDELH=,F6.2)
   40 FORMAT(5X,32HINDICES FOR ATOMS OF FRAGMENT A:)
   41 FORMAT(5X,32HINDICES FOR ATOMS OF FRAGMENT B:)
   42 FORMAT(5X,43HDISTANCE BETWEEN THESE ATOMS DEFINES R.C. :,2I4)
   43 FORMAT(3X,36HNUMBER OF ADDITIONAL REACTION PATHS=,I2)
   44 FORMAT(3X,25HPARAMETERS FOR REACTANT A)
   45 FORMAT(5X,28HNORMAL MODE QUANTUM NUMBERS:,20F5.2)
   46 FORMAT(1H ,4X,22HREACTANT IS A DIATOMIC)
   47 FORMAT(1H ,7X,6HNHARM:,I2,5X,6HINDEX:,I3,5X,2HN:,f5.2,2X,2HJ:
     >,f5.3/)
   48 FORMAT(3X,23HINTERNUCLEAR PARAMETERS,/,5X,7HJ-ATOM=,I3,4X,
     *7HK-ATOM=,I3,4X,5HRMAX=,F6.2,4X,5HRBAR=,F6.2,4X,5HDELH=,F8.3)
  900 FORMAT(5X,43HMOMENTS OF INERTIA IX, IY AND IZ(AMU-A**2):,3F8.3)
  901 FORMAT(1H ,2X,25HPARAMETERS FOR REACTANT B)
  902 FORMAT(5X,22HRELATIVE ENERGY(KCAL):,F7.2,5X,22HINITIAL SEPARATION(
     *A):,F6.2)
CXCC
 1902 FORMAT(5X,22HTRANSLATIONAL TEMP.  :,F7.2,5X,22HINITIAL SEPARATION(
     *A):,F6.2)
CXCC           
  903 FORMAT(5X,4HNOB=,I2,5X,8HBMAX(A)=,F5.1)
  904 FORMAT(5X,5HNROT=,I2,5X,5HTROT=,F9.2)
  905 FORMAT(3X,18HCUBIC BETA:   CM1=,F10.6,7H   CM2=,F10.6,7H   CM3=,
     *F10.6,7H   CM4=,F10.6)
  906 FORMAT(5X,30HEQUILIBRIUM COORDINATES FOR A:)
  907 FORMAT(5X,30HEQUILIBRIUM COORDINATES FOR B:)
  908 FORMAT(2F10.0,2I5,F10.0)
  909 FORMAT(6X,25I4)
CXCCC
CX910 FORMAT(10X,26HREACTION OCCURRED FOR PATH,I3)
  910 FORMAT(10X,26HBARRIER  PASSED   FOR PATH,I3,' CYCLE COUNT ',I5)
CXCCC
  911 FORMAT(5I5,3D16.8)
  912 FORMAT(I10,3F10.0)
  914 FORMAT(3X,34HACTIVATE WITH NORMAL MODE SAMPLING,/)
  915 FORMAT(5X,21HREACTANT IS NONLINEAR)
  916 FORMAT(5X,18HREACTANT IS LINEAR)
  918 FORMAT(18A4)
  919 FORMAT(1H ,18A4)
  920 FORMAT(5X,7HHSCALE=,F8.3,9H  PSCALE=,F5.2)
  921 FORMAT(5X,7HHSCALE=,F8.3)
  922 FORMAT(6I5,3F10.0)
  923 FORMAT(3X,3HNI=,I4,5H  NJ=,I4,5H  NK=,I4,5H  NL=,I4,5H  NM=,
     *I4,5H  NN=,I4,6H  GC1=,F10.5,6H  GQ1=,D15.7,6H  GQ2=,D15.7)
  924 FORMAT(F10.0,I5)
  925 FORMAT(3X,5HHINC=,F9.6,5X,5HNPTS=,I2,/)
  926 FORMAT(3X,3HNI=,I4,5H  NJ=,I4,5H  NK=,I4,5H  NL=,I4,
     *5H  NM=,I4)
  927 FORMAT(3X,47HQUADRATIC FORCE CONSTANTS -- EQUILIBRIUM VALUES)
  928 FORMAT(5X,4HFIJ=,F7.4,5H FIK=,F7.4,5H FIL=,F7.4,5H FJK=,F7.4,
     *5H FJL=,F7.4,5H FKL=,F7.4)
  929 FORMAT(3X,46HQUADRATIC FORCE CONSTANTS -- ASYMPTOTIC VALUES)
  930 FORMAT(3X,43HCUBIC FORCE CONSTANTS -- EQUILIBRIUM VALUES)
  931 FORMAT(5X,4HGIJ=,F7.4,5H GIK=,F7.4,5H GIL=,F7.4,5H GJK=,F7.4,
     *5H GJL=,F7.4,5H GKL=,F7.4)
  932 FORMAT(3X,42HCUBIC FORCE CONSTANTS -- ASYMPTOTIC VALUES)
  933 FORMAT(3X,45HQUARTIC FORCE CONSTANTS -- EQUILIBRIUM VALUES)
  934 FORMAT(5X,4HHIJ=,F7.4,5H HIK=,F7.4,5H HIL=,F7.4,5H HJK=,F7.4,
     *5H HJL=,F7.4,5H HKL=,F7.4)
  935 FORMAT(3X,44HQUARTIC FORCE CONSTANTS -- ASYMPTOTIC VALUES)
  936 FORMAT(3X,24HEQUILIBRIUM ANGLE VALUES)
  937 FORMAT(5X,5HTHIJ=,F8.4,6H THIK=,F8.4,6H THIL=,F8.4,
     *6H THJK=,F8.4,6H THJL=,F8.4,6H THKL=,F8.4)
  938 FORMAT(3X,34HASYMPTOTIC ANGLE VALUES -- SET ONE)
  939 FORMAT(3X,34HASYMPTOTIC ANGLE VALUES -- SET TWO)
  940 FORMAT(3X,35HOUT-OF-PLANE QUADRATIC AND QUARTIC
     *,15HFORCE CONSTANTS)
  941 FORMAT(5X,3HF1=,F7.4,5H  F2=,F7.4,5H  F3=,F7.4,
     *5H  F4=,F7.4)
  942 FORMAT(5X,3HH1=,F7.4,5H  H2=,F7.4,5H  H3=,F7.4,
     *5H  H4=,F7.4)
  943 FORMAT(3X,34HNON-DIAGONAL CUBIC FORCE CONSTANTS)
  944 FORMAT(5X,4HGN1=,F7.4,5H GN2=,F7.4,5H GN3=,F7.4,
     *5H GN4=,F7.4,5H GN5=,F7.4)
  945 FORMAT(3X,24HEQUILIBRIUM BOND LENGTHS)
  946 FORMAT(5X,3HR1=,F12.8,5H  R2=,F12.8,5H  R3=,F12.8,
     *5H  R4=,F12.8)
  947 FORMAT(3X,6HNFTET=,I2,4X,7HNUMTET=,I2)
  948 FORMAT(15X,43HINDICES OF TETRAHEDRAL ANGLES TO BE PRINTED)
  949 FORMAT(3X,3HNI=,I4,5H  NJ=,I4,5H  NK=,I4,5H  NL=,I4,6H FRRZ=,F9.5,
     *6H RIJ0=,F8.4,5H CIJ=,F6.3,6H RKL0=,F8.4,5H CKL=,F6.3)
  950 FORMAT(3X,3HNI=,I4,5H  NJ=,I4,5H  NB=,I4,5H NRT=,I4,6H  FRT=,F9.5,
     *7H  R110=,F8.4,5H CIJ=,F6.3)
  951 FORMAT(3X,14HDIHEDRAL ANGLE,I3,6H,  NI=,I4,
     *5H  NJ=,I4,5H  NK=,I4,5H  NL=,I4)
  952 FORMAT(10X,I2,6H  FDH=,F7.3,6H  GDH=,F8.3)
  953 FORMAT(3X,5HNFDH=,I2,5X,6HNUMDH=,I2)
  954 FORMAT(15X,40HINDICES OF DIHEDRAL ANGLES TO BE PRINTED)
  955 FORMAT(3X,35HACTIVATE WITH LOCAL MODE EXCITATION,/)
  956 FORMAT(3X,36HPARAMETERS FOR LOCAL MODE EXCITATION,/,10X,
     *       9HBOND I,J=,2I3,21H   LOCAL MODE ENERGY=,F8.3,/,9X,
     *       16H EDELTA(BOXING)=,F6.3,/)
  957 FORMAT(10X,'BOXING (N+1) AND (N-1) LEVELS',/,
     *10X,'E(N+1)=',F10.3,2X,'EDEL=',F8.3,/,
     *10X,'E(N-1)=',F10.3,2X,'EDEL=',F8.3)
  958 FORMAT(3X,6HMPLOT=,I2,8H  NPLOT=,I3,6H  NLM=,I3,7H  MNTR=,I3)
  959 FORMAT(3X,6H NLMI=,I2,6H NLMJ=,I2,6H  RLZ=,F6.3)
  960 FORMAT(3X,50H**  VZ LESS THAN ZERO INDICATES 3-FOLD TORSION  **)
 1000 FORMAT(2I10,F15.0)
 1001 FORMAT(3X,5H  NB=,I4,5H NBB=,I4,6H  FTT=,F9.5)
 1904 FORMAT(3X,' Diatom is treated classically'/8H   Tvib=,f10.4,
     *6H Trot=,f10.4)
C
 1110 CONTINUE  
C
C         INITIALIZE ARRAYS
C
      iffa=0
      DO 3000 I=1,2000
      NEVIB(I)=0
      NEVIBU(I)=0
      NEVIBL(I)=0
      NEVIBN(I)=0
      ESAV(I)=0.0D0
      ESQ(I)=0.0D0
 3000 CONTINUE
      DO 3001 I=1,25
      NDH(I)=0
 3001 CONTINUE
C
C              INITIALIZE PARAMETERS
C
      NMA=0
C
C         CONSTANTS WITHIN THE COMPUTER PROGRAM
C              BOND ENERGIES FROM KCAL/MOLE:(C1)
C              HARMONIC STRETCH FORCE CONSTANT FROM MDYN/A:(C2)
C              HARMONIC BEND FORCE CONSTANT FROM MDYN-A/RAD**2:(C3)
C              EQUILIBRIUM ANGLES FROM DEGREES:(C4)
C              GAS LAW CONSTANT IN INTEGRATION UNITS:(C5)
C              FREQUENCIES(X TWOPI) FROM CM-1:(C6)
C              PLANCK'S CONSTANT DIVIDED BY TWOPI IN INTEGRATION UNITS:(C7)
C
      C1=0.04184000D0
      C2=6.022045D0
      C3=6.022045D0
      C4=0.01745329D0
      C5=0.083144D-3
      C6=1.8836518D-3
      C7=0.063508D0
      RAA1=1.0D0/2.0D0
      RA1=1.0D0- SQRT(2.0D0)/2.0D0
      RA2=2.0D0*RA1
      RA3=2.0D0-3.0D0* SQRT(2.0D0)/2.0D0
      RB1=2.0D0-RA1
      RB2=2.0D0*RB1
      RB3=4.0D0-RA3
      RC1=1.0D0/6.0D0
      RC2=1.0D0/3.0D0
      PI=3.141592653589793D0
C     PI=4.0D0*DATAN(1.0D0)
      HALFPI=PI/2.0D0
      TWOPI=2.0D0*PI
CXP INPUT SECTION lasts till endofinput
C
C        READ AND WRITE TWO TITLE CARDS
C
CXP
      IFINISH=0
c     READ(5,918)(TITLE(I),I=1,36)
CXCCa
 1903 FORMAT(A40)
      READ(5,1903) FNAM1
      read(5,1903) FNAM2
      read(5,1903) FNAM3

CXP    prevent reading if end-of-file; go to stop other PE's
c      IF(FNAM1 .EQ. 'cccccc')   stop 5
      OPEN(9,FILE=FNAM1)
      open(12,file=FNAM2)
      open(120,file=FNAM3)
CXP
CXP   READ(5,1903) FNAM2
CXP   OPEN(10,FILE=FNAM2)
cx    READ(5,1903) FNAM3
cx    OPEN(6,FILE=FNAM3)
cx    WRITE(6,1919)
cx    WRITE(6,919)(TITLE(I),I=1,36)
cx    WRITE(9,1919)
cx    WRITE(9,919)(TITLE(I),I=1,36)
CXP   WRITE(10,1919)
CXP   WRITE(10,919)(TITLE(I),I=1,36)
 1919 FORMAT(1H ,' venusozone: unparallelized version of VENUS'
     *,' for polyatomic applications'/'Pascal Honvault PES for H+O2')
CXCCz
C
C        READ # OF ATOMS AND # OF POTENTIAL TYPES
C
      READ(5,*)NATOMS,NST,NM,NB,NA,NLJ,NTAU,NEXP,NGHOST,NTET,
     *NVRR,NVRT,NVTT,NANG
      WRITE(6,14)NATOMS,NST,NM,NB,NA,NLJ,NTAU,NEXP,NGHOST,NTET,
     *NVRR,NVRT,NVTT,NANG
      I3N=3*NATOMS
C
C        READ POTENTIAL PARAMETERS
C

      IF(NST.EQ.0)GO TO 51
      DO 50 I=1,NST
      READ(5,2)N1J(I),N1K(I),RSZ(I),FS(I)
      IF(RSZ(I).EQ.0.0D0) THEN
      RSZ(I)=RSZ(I-1)
      FS(I)=FS(I-1)
      ENDIF
      WRITE(6,3)N1J(I),N1K(I),RSZ(I),FS(I)
   50 CONTINUE
      WRITE (6,15)
   51 IF (NM.EQ.0) GO TO 53
      DO 52 I=1,NM
      READ(5,2)N2J(I),N2K(I),RMZ(I),B(I),D(I)
      IF(RMZ(I).NE.0.0D0)GO TO 1052
      RMZ(I)=RMZ(I-1)
      B(I)=B(I-1)
      D(I)=D(I-1)
 1052 CONTINUE
      WRITE(6,4)N2J(I),N2K(I),RMZ(I),B(I),D(I)
      IF(B(I).GT.0.0D0)GO TO 52
      READ(5,7)CM1(I),CM2(I),CM3(I),CM4(I)
      WRITE(6,905)CM1(I),CM2(I),CM3(I),CM4(I)
   52 CONTINUE
      WRITE (6,15)
   53 IF (NB.EQ.0) GO TO 55
      DO 54 I=1,NB
      READ(5,5)N3J(I),N3K(I),N3M(I),THETAZ(I),FBZ(I),RJZ(I),CJ(I),RKZ(I)
     *,CK(I)
      IF(THETAZ(I).EQ.0.0D0) THEN
      THETAZ(I)=THETAZ(I-1)
      FBZ(I)=FBZ(I-1)
      RJZ(I)=RJZ(I-1)
      CJ(I)=CJ(I-1)
      RKZ(I)=RKZ(I-1)
      CK(I)=CK(I-1)
      ENDIF
      WRITE(6,6)N3J(I),N3K(I),N3M(I),THETAZ(I),FBZ(I),RJZ(I),
     *CJ(I),RKZ(I),CK(I)
   54 CONTINUE
      WRITE(6,15)
   55 IF(NA.EQ.0)GO TO 57
      DO 56 I=1,NA
CXAIX   input modified: numbers of atoms involved in the triatomic PES
CX    READ(5,29)N4J(I),N4K(I),N4M(I),N4N(I),FA(I)
cx 29 FORMAT(4I10,5F15.0)
      READ(5,29)N4J(I),N4K(I),N4M(I)
      WRITE(6,30)N4J(I),N4K(I),N4M(I)
   56 CONTINUE
      WRITE(6,15)
   57 IF(NLJ.EQ.0)GO TO 59
      DO 58 I=1,NLJ
      READ(5,911)N5J(I),N5K(I),NREP(I),MREP(I),LREP(I),ALJ(I),BLJ(I),
     *CLJ(I)
CXP   IF(NREP(I).NE.0.OR.MREP(I).NE.0)GO TO 58
      IF(NREP(I).EQ.0.AND.MREP(I).EQ.0) THEN
      NREP(I)=NREP(I-1)
      MREP(I)=MREP(I-1)
      LREP(I)=LREP(I-1)
      ALJ(I)=ALJ(I-1)
      BLJ(I)=BLJ(I-1)
      CLJ(I)=CLJ(I-1)
      ENDIF
      WRITE(6,31)N5J(I),N5K(I),ALJ(I),BLJ(I),CLJ(I),NREP(I),MREP(I),
     *LREP(I)
   58 CONTINUE
      WRITE(6,15)
   59 IF(NTAU.EQ.0)GO TO 61
      WRITE(6,960)
      DO 60 I=1,NTAU
      READ(5,33)N6I(I),N6J(I),N6K(I),N6L(I),N6M(I),N6N(I),VZTAU(I)
      WRITE(6,34)N6I(I),N6J(I),N6K(I),N6L(I),N6M(I),N6N(I),VZTAU(I)
   60 CONTINUE
      WRITE(6,15)
   61 IF(NEXP.EQ.0)GO TO 63
      DO 62 I=1,NEXP
      READ(5,5)N7J(I),N7K(I),NPOW(I),AEX(I),BEX(I),CEX(I)
CXP   IF(NPOW(I).NE.0)GO TO 62
      IF(NPOW(I).EQ.0) THEN
      NPOW(I)=NPOW(I-1)
      AEX(I)=AEX(I-1)
      BEX(I)=BEX(I-1)
      CEX(I)=CEX(I-1)
      ENDIF
      WRITE(6,37)N7J(I),N7K(I),NPOW(I),AEX(I),BEX(I),CEX(I)
   62 CONTINUE
      WRITE(6,15)
   63 IF(NGHOST.EQ.0)GO TO 65
      DO 64 I=1,NGHOST
      READ(5,922)N8I(I),N8J(I),N8K(I),N8L(I),N8M(I),N8N(I),
     *GC1(I),GEX1(I),GEX2(I)
      WRITE(6,923)N8I(I),N8J(I),N8K(I),N8L(I),N8M(I),N8N(I),
     *GC1(I),GEX1(I),GEX2(I)
   64 CONTINUE
      WRITE(6,15)
   65 IF(NTET.EQ.0) GO TO 67
      DO 66 I=1,NTET
      READ(5,8)N9I(I),N9J(I),N9K(I),N9L(I),N9M(I)
      READ(5,7)(FT0(I,J),J=1,6)
      READ(5,7)(FT2(I,J),J=1,6)
      READ(5,7)(GT0(I,J),J=1,6)
      READ(5,7)(GT2(I,J),J=1,6)
      READ(5,7)(HT0(I,J),J=1,6)
      READ(5,7)(HT2(I,J),J=1,6)
      READ(5,7)(THT(I,J),J=1,6)
      READ(5,7)(THT1(I,J),J=1,6)
      READ(5,7)(THT2(I,J),J=1,6)
      READ(5,7)(FD1(I,J),J=1,4)
      READ(5,7)(HD1(I,J),J=1,4)
      READ(5,7)(GN0(I,J),J=1,5)
      READ(5,7)(R0(I,J),J=1,4)
      WRITE(6,926)N9I(I),N9J(I),N9K(I),N9L(I),N9M(I)
      WRITE(6,927)
      WRITE(6,928)(FT0(I,J),J=1,6)
      WRITE(6,929)
      WRITE(6,928)(FT2(I,J),J=1,6)
      WRITE(6,930)
      WRITE(6,931)(GT0(I,J),J=1,6)
      WRITE(6,932)
      WRITE(6,931)(GT2(I,J),J=1,6)
      WRITE(6,933)
      WRITE(6,934)(HT0(I,J),J=1,6)
      WRITE(6,935)
      WRITE(6,934)(HT2(I,J),J=1,6)
      WRITE(6,936)
      WRITE(6,937)(THT(I,J),J=1,6)
      WRITE(6,938)
      WRITE(6,937)(THT1(I,J),J=1,6)
      WRITE(6,939)
      WRITE(6,937)(THT2(I,J),J=1,6)
      WRITE(6,940)
      WRITE(6,941)(FD1(I,J),J=1,4)
      WRITE(6,942)(HD1(I,J),J=1,4)
      WRITE(6,943)
      WRITE(6,944)(GN0(I,J),J=1,5)
      WRITE(6,945)
      WRITE(6,946)(R0(I,J),J=1,4)
   66 CONTINUE
      WRITE(6,15)
   67 IF(NVRR.EQ.0)GO TO 70
      DO 69 I=1,NVRR
      READ(5,29)N10I(I),N10J(I),N10K(I),N10L(I),
     *FKRRZ(I),RIJ0(I),CIJ(I),RKL0(I),CKL(I)
      WRITE(6,949)N10I(I),N10J(I),N10K(I),N10L(I),
     *FKRRZ(I),RIJ0(I),CIJ(I),RKL0(I),CKL(I)
   69 CONTINUE
      WRITE(6,15)
   70 IF(NVRT.EQ.0)GO TO 72
      DO 71 I=1,NVRT
      READ(5,29)N11I(I),N11J(I),N11B(I),NRT(I),
     *FKRTZ(I),R110(I),CRT(I)
      WRITE(6,950)N11I(I),N11J(I),N11B(I),NRT(I),
     *FKRTZ(I),R110(I),CRT(I)
   71 CONTINUE
      WRITE(6,15)
   72 IF(NVTT.EQ.0)GO TO 76
      DO 75 I=1,NVTT
      READ(5,1000)N12B(I),N12BB(I),FKTTZ(I)
      WRITE(6,1001)N12B(I),N12BB(I),FKTTZ(I)
   75 CONTINUE
      WRITE(6,15)
   76 IF(NANG.EQ.0) GO TO 74
      DO 73 I=1,NANG
      READ(5,8)N12I(I),N12J(I),N12K(I),N12L(I)
      READ(5,8) NDH(I)
      K=NDH(I)
      WRITE(6,951)I,N12I(I),N12J(I),N12K(I),N12L(I)
      DO 73 J=1,K
      READ(5,7) FDH(I,J),GDH(I,J)
      WRITE(6,952) J,FDH(I,J),GDH(I,J)
   73 CONTINUE
      WRITE(6,15)
   74 CONTINUE
C
C        CONVERT POTENTIAL PARAMETERS TO INTEGRATION UNITS
C
      DO 80 I=1,NST
   80 FS(I)=FS(I)*C2
      DO 81 I=1,NM
   81 D(I)=D(I)*C1
      DO 82 I=1,NB
      FBZ(I)=FBZ(I)*C3
   82 THETAZ(I)=THETAZ(I)*C4
      DO 83 I=1,NA
   83 FA(I)=FA(I)*C3
      DO 84 I=1,NLJ
      ALJ(I)=ALJ(I)*C1
      BLJ(I)=BLJ(I)*C1
   84 CLJ(I)=CLJ(I)*C1
      DO 86 I=1,NTAU
   86 VZTAU(I)=VZTAU(I)*C1
      DO 87 I=1,NEXP
      AEX(I)=AEX(I)*C1
   87 CEX(I)=CEX(I)*C1
      DO 88 I=1,NGHOST
      GEX1(I)=GEX1(I)*C1
   88 GEX2(I)=GEX2(I)*C1
      DO 93 I=1,NTET
      DO 89 J=1,6
      FT0(I,J)=FT0(I,J)*C3
      FT2(I,J)=FT2(I,J)*C3
      GT0(I,J)=GT0(I,J)*C3
      GT2(I,J)=GT2(I,J)*C3
      HT0(I,J)=HT0(I,J)*C3
      HT2(I,J)=HT2(I,J)*C3
      THT(I,J)=THT(I,J)*C4
      THT1(I,J)=THT1(I,J)*C4
      THT2(I,J)=THT2(I,J)*C4
   89 CONTINUE
      DO 91 J=1,4
      FD1(I,J)=FD1(I,J)*C3
      HD1(I,J)=HD1(I,J)*C3
   91 CONTINUE
      DO 92 J=1,5
      GN0(I,J)=GN0(I,J)*C3
   92 CONTINUE
   93 CONTINUE
      DO 94 I=1,NVRR
   94 FKRRZ(I)=FKRRZ(I)*C2
      DO 95 I=1,NVRT
   95 FKRTZ(I)=FKRTZ(I)*C2
      DO 1097 I=1,NVTT
 1097 FKTTZ(I)=FKTTZ(I)*C3
      DO 1096 I=1,NANG
      K=NDH(I)
      DO 1096 J=1,K
      FDH(I,J)=FDH(I,J)*C1
      GDH(I,J)=GDH(I,J)*C4
 1096 CONTINUE
C
      READ(5,*)VZERO
      WRITE(6,38)VZERO
      VZERO=VZERO*C1
C
C        INPUT FOR TRAJECTORIES
C
C            NSELT=-2 PROGRAM FINDS REACTION PATH
C            NSELT=-1 PROGRAM DOES NORMAL MODE ANALYSIS
C            NSELT=0  INITIAL Q'S AND P'S ARE READ IN
C            NSELT=1  PROGRAM FINDS MINIMUM ENERGY GEOMETRY
C                     (INITIAL Q'S AND P'S ARE READ IN)
C            NSELT=2  CHOOSE INITIAL CONDITIONS FOR ONE OR TWO REACTANTS,
C                     NACT ARE THE OPTIONS FOR CHOOSING INITIAL CONDITIONS.
C            NACT=0   REACTANTS ARE ATOMS AND/OR DIATOMS.
C            NACT=1   ACTIVATE WITH ORTHANT SAMPLING
C            NACT=2   ACTIVATE WITH MICROCANONICAL NORMAL MODE SAMPLING
C            NACT=3   ACTIVATE WITH FIXED NORMAL MODE ENERGIES
C            NACT=4   LOCAL MODE EXCITATION
C
C        INITIALIZE SOME PARAMETERS.  SPECIFIC VALUES MAY BE READ IN.
C
      NATOMA(1)=0
      NATOMB(1)=0
      NT=1
      NPATHS=0
      NABJ(1)=1
      NABK(1)=1
CXpoly
      NABL(1)=0
      RMAX(1)=100.D0
      RBAR(1)=100.D0
C
CXCC  READ(5,8)NSELT,NACT
      READ(5,998)NSELT,NACT,LSEC,blinit
 998  FORMAT(3I10,f20.10)
CXCC
      READ(5,7)(W(I),I=1,NATOMS)
      WRITE (6,15)
      WRITE(6,28)NATOMS
      WRITE(6,10)(W(I),I=1,NATOMS)
      WRITE(6,15)
      IF(NSELT.LT.0)GO TO 97
cx  read NT,NS,NIP,NCROT in free format
      READ(5,*)NT,NS,NIP,NCROT
      WRITE(6,*)NT,NS,NIP,NCROT
      READ(5,9)KRAND,TIME
C       Initialize random number generation
cx  www.homepages.ucl.ac.uk/~ucakarc/work/randgen.html 
      CALL ZBQLINI(KRAND)
      WRITE(6,12)KRAND,TIME
      IF(NSELT.EQ.2)GO TO 98
      WRITE(6,27)NSELT
   97 CONTINUE
      IF(NSELT.EQ.0)GO TO 140
      IF(NSELT.EQ.1)GO TO 145
   98 IF(NSELT.EQ.2.AND.NACT.LE.1)GO TO 99
C
C        READ DISPLACEMENT INTERVAL, HINC.
C        READ NUMBER OF DISPLACEMENTS ABOUT CARTESIAN MINIMUM, NPTS=1 OR 2.
C
      READ(5,924)HINC,NPTS
      WRITE(6,925)HINC,NPTS
      IF(NSELT.EQ.-1)GO TO 145
      IF(NSELT.EQ.-2)GO TO 96
   99 CONTINUE
      WRITE(6,26)
      IF(NACT.EQ.1) WRITE(6,13)
      IF(NACT.EQ.2) WRITE(6,35)
      IF(NACT.EQ.3) WRITE(6,914)
      IF(NACT.EQ.4) WRITE(6,955)
C
C         READ PARAMETERS FOR REACTANTS
C
C
C          ***PARAMETERS FOR REACTANT A***
C
   96 READ(5,8)NATOMA(1),NLINA
C
C             NLINA=0, MOLECULE IS NONLINEAR
C             NLINA=1, MOLECULE IS LINEAR
C
      K=NATOMA(1)
      WRITE(6,44)
      WTA(1)=0.0D0
      DO 100 J=1,K
      LA(1,J)=J
      LL(J)=LA(1,J)
  100 WTA(1)=WTA(1)+W(J)
      IF(NSELT.EQ.-2)GO TO 107
      K=3*K
      READ(5,7)(QZA(1,J),J=1,K)
      WRITE(6,906)
      WRITE(6,36)(QZA(1,J),J=1,K)
C
C             TRANSFORM QZA TO CENTER OF MASS FRAME
C
      DO 108 J=1,K
      Q(J)=QZA(1,J)
  108 CONTINUE
      WT=WTA(1)
      N=NATOMA(1)

      CALL CENMAS(WT,QCM,VCM,N)

      DO 106 J=1,K
      QZA(1,J)=QQ(J)
  106 CONTINUE
C
C             PARAMETERS FOR DIATOM
C             IF NHARMA=0, DIATOM IS TREATED AS A HARMONIC OSCILLATOR.
C             INDEX IS THE INDEX FOR THE DIATOM STRETCH, WHICH IS EITHER
C             A HARMONIC OR MORSE OSCILLATOR.
CXAIX   if NHARMA is negative, the diatom energies are selected from a thermal
CXAIX     ensemble with a fixed vibr. and rot. temperature
C
      IF(NATOMA(1).GT.2)GO TO 103
      IF(NATOMA(1).EQ.1)GO TO 107
CXCCa READ(5,8)NHARMA,INDEX,NNA,JA
      READ(5,29)NHARMA,INDEX,NNA,JA,TVIBA,TROTA
CXCCz
      WRITE(6,46)
      WRITE(6,47)NHARMA,INDEX,NNA,JA
CXCCa
      IF(NHARMA .LT. 0) WRITE(6,1904) TVIBA,TROTA 
CXCC  IF(NHARMA.NE.0)GO TO 102
      IF(NHARMA.NE.0 .OR. NHARMA .NE. -1)GO TO 102
CXCCz
      RZA=RSZ(INDEX)
      FAHARM=FS(INDEX)
      GO TO 107
  102 RZA=RMZ(INDEX)
      BA=B(INDEX)
      DA=D(INDEX)
      GO TO 107
C
C             REACTANT A IS A POLYATOMIC
C
  103 CONTINUE
      IF(NACT.EQ.3 .OR. NACT.EQ.4) GOTO 104
      READ(5,7)ENMTA,PSCALA
      IF(NACT.EQ.1) WRITE(6,920)ENMTA,PSCALA
      IF(NACT.EQ.2) WRITE(6,921)ENMTA
      GO TO 105
C
  104 CONTINUE
      J=K-6+NLINA
      READ(5,7)(ANQA(I),I=1,J)
      WRITE(6,45)(ANQA(I),I=1,J)
C
C             SET NMA AND INITIALIZE N.M. ARRAYS
C
CXP      NMA=J
C
CXP      DO 1440 I=1,NMA
CXP      DO 1440 J=1,2000
CXP      ENSAV(I,J)=0.0D0
CXP      ENSQ(I,J)=0.0D0
CXP 1440 CONTINUE
C
C             NROTA=0, SAMPLE ROTATIONAL ENERGY FROM A THERMAL
C                      DISTRIBUTION
C             NROTA=1, ROTATIONAL ENERGY OF EACH AXIS IS RT/2
C
  105 CONTINUE
      READ(5,912)NROTA,TROTA
      WRITE(6,904)NROTA,TROTA
      READ(5,7)(AI(I),I=1,3)
      WRITE(6,900)(AI(I),I=1,3)
C
C         PARAMETERS FOR LOCAL MODE EXCITATION
C         NONI    : ATOM I
C         NONJ    : ATOM J
C         ENON    : THE INITIAL BOND ENERGY IN KCAL/MOLE
C         EDELTA  : BOXING LENGTH OF ENERGY
C
      IF(NACT.NE.4) GO TO 109
      WRITE(6,15)
      READ(5,2)NONI,NONJ,ENON,EDELTA
      WRITE(6,956)NONI,NONJ,ENON,EDELTA
      READ(5,7)ENU,EDELTU,ENL,EDELTL
      WRITE(6,957)ENU,EDELTU,ENL,EDELTL
  109 CONTINUE
C
C         PARAMETERS FOR PLOTTING MODE ENERGIES
C             NPLOT   : THE INCREMENT IN CYCLE COUNT FOR LISTING MODE
C                       ENERGIES
C             MPLOT   : FOR THE TIME AVERAGED PLOTTING
C                       MPLOT=0 : DO NOT PRINT ON FILE 12 THE AVERAGE
C                                 MODE ENERGIES AT CYCLES
C                       MPLOT=1 : PRINT ON FILE 12 THE AVERAGE MODE
C                                 ENERGIES AT CYCLES INCREMENTED BY
C                                 NPLOT
C             NLM     : NUMBER OF BOND MODES MONITORED IN ADDITION TO
C                       THE EXCITED ONE
C             MNTR    : INDEX FOR NORMAL MODE WHOSE POPULATION IS TO BE
C                       MONITORED
C
      READ (5,8)MPLOT,NPLOT,NLM,MNTR
      WRITE (6,958)MPLOT,NPLOT,NLM,MNTR
CXP      IF(MPLOT.NE.1.OR.NLM.EQ.0) GO TO 1410
C
C          INITIALIZE THE L.M. ARRAYS EXCEPT THE EXCITED ONE
C
CXP      DO 1415 I=1,NLM
CXP      DO 1415 J=1,2000
CXP      EBSAV(I,J)=0.0D0
CXP      EBSQ(I,J)=0.0D0
CXP 1415 CONTINUE
CXP 1410 CONTINUE
C
C          ***PARAMETERS FOR REACTANT B***
C
  107 READ(5,8)NATOMB(1),NLINB
      IF(NATOMB(1).EQ.0.AND.NSELT.EQ.-2)GO TO 145
      IF(NATOMB(1).EQ.0)GO TO 140
      WRITE(6,15)
      WRITE(6,15)
      WRITE(6,901)
      K=NATOMB(1)
      WTB(1)=0.0D0

C**********
      wtbsz(1)=1.0d0
C**********

      DO 120 J=1,K
      M=J+NATOMA(1)
      LB(1,J)=M
      LL(J)=LB(1,J)
      WTB(1)=WTB(1)+W(M)

C***********
  120 wtbsz(1)=wtbsz(1)*w(m)
      redmassb(1)=wtbsz(1)/wtb(1)
C***********

      IF(NSELT.EQ.-2)GO TO 145
      K=3*K
cx6SEC  may be better to set QZB=0 and QZB(3)=RMZ(1)
      READ(5,7)(QZB(1,J),J=1,K)
      WRITE(6,907)
      WRITE(6,36)(QZB(1,J),J=1,K)
C
C             TRANSFORM QZB TO CENTER OF MASS FRAME
C
      I=3*NATOMA(1)
      DO 128 J=1,K
      Q(J+I)=QZB(1,J)
  128 CONTINUE
      WT=WTB(1)
      N=NATOMB(1)

      CALL CENMAS(WT,QCM,VCM,N)
      DO 127 J=1,K
      QZB(1,J)=QQ(J+I)
  127 CONTINUE
C
C             PARAMETERS FOR DIATOM
C             IF NHARMB=0, DIATOM IS TREATED AS A HARMONIC OSCILLATOR.
C             INDEX IS THE INDEX FOR THE DIATOM STRETCH, WHICH IS EITHER
C             A HARMONIC OR MORSE OSCILLATOR.
C
      IF(NATOMB(1).GT.2)GO TO 123
      IF(NATOMB(1).EQ.1)GO TO 126
CXCCa READ(5,8)NHARMB,INDEX,NNB,JB
cx   read non-integer quantum numbers
      READ(5,*)NHARMB,INDEX,VNB,RJB
CXCCz
      WRITE(6,41)
      WRITE(6,47)NHARMB,INDEX,VNB,RJB
CXCCa
      IF(NHARMB .LT. 0) WRITE(6,1904) TVIBB,TROTB 
CXCC  IF(NHARMB.NE.0)GO TO 122
      IF(NHARMB.NE.0 .OR. NHARMB .NE. -1)GO TO 122
CXCCz
      RZB=RSZ(INDEX)
      FBHARM=FS(INDEX)
      GO TO 126
  122 RZB=RMZ(INDEX)
      BB=B(INDEX)
      DB=D(INDEX)
      GO TO 126
C
C             REACTANT B IS A POLYATOMIC
C
  123 CONTINUE
      IF(NACT.EQ.3)GO TO 124
      READ(5,7)ENMTB,PSCALB
      IF(NACT.EQ.1) WRITE(6,920)ENMTB,PSCALB
      IF(NACT.EQ.2) WRITE(6,921)ENMTB
      GO TO 125
C
  124 CONTINUE
      J=K-6+NLINB
      READ(5,7)(ANQB(I),I=1,J)
      WRITE(6,45)(ANQB(I),I=1,J)
C
C             NROTB=0, SAMPLE ROTATIONAL ENERGY FROM A THERMAL
C                      DISTRIBUTION
C             NROTB=1, ROTATIONAL ENERGY OF EACH AXIS IS RT/2
C
  125 CONTINUE
      READ(5,912)NROTB,TROTB
      WRITE(6,904)NROTB,TROTB
      READ(5,7)(BI(I),I=1,3)
      WRITE(6,900)(BI(I),I=1,3)
C
C        REACTANT A AND B INTERNUCLEAR PARAMETERS FOR THE FIRST
C        REACTION PATH.
C
  126 CONTINUE
      READ(5,2)NABJ(1),NABK(1),RMAX(1),RBAR(1),DELH(1)
      WRITE(6,48)NABJ(1),NABK(1),RMAX(1),RBAR(1),DELH(1)
C
C             NOB=1, RANDOMLY SAMPLE IMPACT PARAMETER BETWEEN 0 AND BMAX
cx                        quadratically
C             NOB=0, IMPACT PARAMETER EQUALS BMAX
C             NOB=2, RANDOMLY SAMPLE IMPACT PARAMETER BETWEEN 0 AND BMAX
cx                         linearly (importance sampling)
C
CXCCa
      TTRAN=0.D0
      READ(5,*)SEREL,S
      WRITE(6,902)SEREL,S
CXCC  SEREL=SEREL*C1
      IF(LSEC .GE. 2) THEN
      TTRAN=SEREL
      WRITE(6,1902)TTRAN,S
      ELSE 
      WRITE(6,902)SEREL,S
      SERL=SEREL
      SEREL=SEREL*C1
      ENDIF
CXCCz
      READ(5,*)NOB,BMAX
      WRITE(6,903)NOB,BMAX
CXCCa
c      WRITE(9,1901)ENMTA,ENMTB,TTRAN,BMAX,NOB,LSEC,S
 1901 FORMAT(4F18.10/2I5,f18.10)
CXCCz
C
C        PARAMETERS FOR CLASSIFYING REACTION EVENTS
C        NPATHS=NUMBER OF REACTION PATHS IN ADDITION TO THAT FOR REACTANTS
C        REACTION PATH 1 IS FOR REACTANTS A AND B
C
  140 CONTINUE
      WRITE(6,15)
      READ(5,8)NPATHS
      WRITE(6,43)NPATHS
      IF(NPATHS.EQ.0)GO TO 145
      M=NPATHS+1
      DO 144 I=2,M
      READ(5,908)RMAX(I),RBAR(I),NATOMA(I),NATOMB(I),DELH(I)
      WRITE(6,39)I,RMAX(I),RBAR(I),NATOMA(I),NATOMB(I),DELH(I)
CXpolyREAD(5,8)NABJ(I),NABK(I)
CXpolyWRITE(6,42)NABJ(I),NABK(I)
CXpoly  NABL identifies the third atom in channel NPATH=I
      READ(5,8)NABJ(I),NABK(I),NABL(I)
      WRITE(6,42)NABJ(I),NABK(I),NABL(I)
      K=NATOMA(I)
      READ(5,1)(LA(I,J),J=1,K)
      WTA(I)=0.0D0
      DO 142 J=1,K
  142 WTA(I)=WTA(I)+W(LA(I,J))
      WRITE(6,40)
      WRITE(6,909)(LA(I,J),J=1,K)
      K=3*NATOMA(I)
cx6SEC   the first product is monatomic
      QZA(I,1)=0.0
      QZA(I,2)=0.0
      QZA(I,3)=0.0
cx6SECREAD(5,7)(QZA(I,J),J=1,K)
c      READ(5,7)(QZA(I,J),J=1,K)
      WRITE(6,906)
      WRITE(6,36)(QZA(I,J),J=1,K)
      K=NATOMB(I)
      IF(K.EQ.0)GO TO 144
      READ(5,1)(LB(I,J),J=1,K)
      WTB(I)=0.0D0

C***********
      wtbsz(i)=1.0d0
C***********

      DO 143 J=1,K

C***********
      wtbsz(i)=wtbsz(i)*w(lb(i,j))
C***********

  143 WTB(I)=WTB(I)+W(LB(I,J))

C************
      redmassb(i)=wtbsz(i)/wtb(i)
C************
      WRITE(6,41)
      WRITE(6,909)(LB(I,J),J=1,K)
      K=3*NATOMB(I)
cx6SEC  may be better to set QZB=0 and QZB(I,3)=RMZ(x)
      QZB(I,1)=0.0
      QZB(I,2)=0.0
      QZB(I,3)=RMZ(I)
cx6SECREAD(5,7)(QZB(I,J),J=1,K)
c      READ(5,7)(QZB(I,J),J=1,K)
      WRITE(6,907)
      WRITE(6,36)(QZB(I,J),J=1,K)
      WRITE(6,15)
  144 CONTINUE
  145 CONTINUE
cx6SEC  skip reading unused parameters
      go to 207
C
C        INFORMATION TO BE PRINTED
C
      READ(5,8)NFQP
      WRITE(6,16)NFQP
C
C        NFQP=0, DO NOT PRINT Q AND P ARRAYS
C
      READ(5,8)NFR,NUMR
      WRITE(6,17)NFR,NUMR
      IF(NFR.EQ.0)GO TO 201
      READ(5,1)(JR(I),KR(I),I=1,NUMR)
      DO 200 I=1,NUMR
      WRITE(6,18)JR(I),KR(I)
  200 CONTINUE
      WRITE(6,15)
  201 READ(5,8)NFB,NUMB
      WRITE(6,19)NFB,NUMB
      IF(NFB.EQ.0)GO TO 202
      READ(5,1)(IB(I),I=1,NUMB)
      WRITE(6,20)
      WRITE(6,21)(IB(I),I=1,NUMB)
      WRITE(6,15)
  202 READ(5,8)NFA,NUMA
      WRITE(6,22)NFA,NUMA
      IF(NFA.EQ.0)GO TO 203
      READ(5,1)(IA(I),I=1,NUMA)
      WRITE(6,23)
      WRITE(6,21)(IA(I),I=1,NUMA)
      WRITE(6,15)
  203 READ(5,8)NFTAU,NUMTAU
      WRITE(6,24)NFTAU,NUMTAU
      IF(NFTAU.EQ.0)GO TO 204
      READ(5,1)(ITAU(I),I=1,NUMTAU)
      WRITE(6,25)
      WRITE(6,21)(ITAU(I),I=1,NUMTAU)
  204 CONTINUE
      READ(5,8)NFTET,NUMTET
      WRITE(6,947)NFTET,NUMTET
      IF(NFTET.EQ.0)GO TO 205
      READ(5,1)(ITET(I),I=1,NUMTET)
      WRITE(6,948)
      WRITE(6,21)(ITET(I),I=1,NUMTET)
  205 CONTINUE
      READ(5,8) NFDH,NUMDH
      WRITE(6,953) NFDH,NUMDH
      IF(NFDH.EQ.0) GO TO 206
      READ(5,1)(IDH(I),I=1,NUMDH)
      WRITE(6,954)
      WRITE(6,21)(IDH(I),I=1,NUMDH)
  206 CONTINUE
      WRITE(6,15)
  207 CONTINUE

C
C        SET FLAGS AND PARAMETERS
C          DEFINITION OF FLAGS
C             NSFLAG:  FOR CALCULATING QMIN,QMAX, AND PMAX.
C                      IN 'INITQP'.
C             NAM:  FOR CALCULATTING ANGULAR MOMENTUM.
C                   IN 'ROTN', 'INITQP', AND 'ORTHAN'.
C             NAST:  FOR PASSING BARRIER.
C                    IN 'MAIN'.
C             NFINAL:  FOR CALCULATING EROT.
C                      IN 'FINAL'.
C
C
      call flush(6)
      NSFLAG=0

      NAM=0
      ATIME=TIME/1440.0D0
CXP   NI=I3N
      NI=3*NATOMS
      NID=2*NI
      NTZ=0

cx start new trajectory
  451 NTZ=NTZ+1
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
       iTini1=1000000
       iTini2=1000000
       iTini3=1000000

       mini1=0
       mini2=0
       mini3=0
         
       iTfin1=0
       iTfin2=0
       iTfin3=0

       iTtot1=0
       iTtot2=0
       iTtot3=0

       olddiff=0.0d0
       somm=0.0
c@================================================================================
c@ filname creator:
c@      WRITE(6,*) ' animate traj. No ',NTZ
cxanim create filename animr_NTZ_FNAM1 animx_NTZ_FNAM1
cxanim unit 72 is the RH1F-RH1H2 plot
c@    FNAME2='animr_E7p_v0j00_1p0eV_bmax_3p0.dat'
cxanim unit 73 is the x,y,z plot for MOLDEN
c@    FNAME3='animX_e7p_v0j00_1p0eV_bmax_3p0.xyz'
c@    ix=NTZ/10
c@    i4=NTZ-10*ix
c@    ic=ix/10
c@    i3=ix-10*ic
c@    id=ic/10
c@    i2=ic-10*id
c@    i1=id/10
c@    fne2(6)=char(i1+48)
c@    fne2(7)=char(i2+48)
c@    fne2(8)=char(i3+48)
c@    fne2(9)=char(i4+48)
c@    fne3(6)=char(i1+48)
c@    fne3(7)=char(i2+48)
c@    fne3(8)=char(i3+48)
c@    fne3(9)=char(i4+48)
c@    write(6,*) FNAME2
c@    write(6,*) FNAME3
c@    OPEN(72,file=FNAME2)
c@    OPEN(73,file=FNAME3)
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
c@
c@  az fnam2 es fnam3 neveket a 'animr E7p30v0.dat' mintara testreszabottan
c@  kell megadni, hogy tudni lehessen utolag, hogy mi van benne (nem eleg, hogy az
c@  x könyvtarban van). A dimenziokat szukség szerint lehet modositani, de 40
c@  krakter altalaban elegendo. Az xx unitszamot az open(xx, utasitasban ugy kell
c@  megadni, hogy az egyezzen azzal, amit a GWRITE-ban hasznalunk kiiratasra.
c@
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

c     write(43,*) 'energy conservation, traj #', NTZ
c      WRITE(6,*) 'NTZ increased',NTZ
CXCCa IF(NTZ.GT.NT)CALL EXIT
      IF(NTZ.GT.NT) THEN      
CXP    get cpu time (real time at the end of the run a PE)
CXUP  WRITE(9,*) 'time used',timef-time0
CXUPCXW...WRITE(6,*) 'time used',timef-time0
      WRITE(6,*) '# of failed trajectories'
      WRITE(6,*) (ifai(iffb),iffb=1,iffa)
        CLOSE(6)
        GO TO 1111
      end if
C!!!!!
c     close(49)
c     fn49='shev'
c     fn(5)=char(NTZ+48)
c     open(49,file=fn49)
      CLOSEST=500.D0
      DELKINO=0.d0
      RELKINO=1.d1
      RTIME1=0.d0
      RTIME2=0.d0
      RTIME3=0.d0
      RTIME4=0.d0

CXCCz
      NC=0
      KRANOD =KRAND
      VRELO=0.0D0
CXCCa
      RCMO=100.D0
CXCCz
      INTST=0
      NAST=2
      NFINAL=0
      KRE=1

C
C          INITIALIZE THE Q AND P ARRAYS
C

      J=3*NATOMA(1)
      DO 453 I=1,J
      Q(I)=QZA(1,I)
  453 P(I)=0.0D0
      K=3*NATOMB(1)
      DO 454 I=1,K
      Q(J+I)=QZB(1,I)
  454 P(J+I)=0.0D0
C          SELECT INITIAL CONDITIONS
CXCCa CALL SELECT
      CALL SELECT(LSEC,TTRAN,blinit)
      CALL DVDQ


      CALL ENERGY
c     write(93,*) NTZ
c     write(93,*) (Q(i),i=1,9),(p(i),i=1,9)

      NPATH=1
      CALL FINAL
      TOTIN=H
      CALL GFINNL
      SER1=ERELSQ
      HINTA=EA(3)
      HINTB=EB(3)
      ERTA0=EROTA
      ERTB0=EROTB
CXW...WRITE(6,*) 'HINTA,EROTA,HINTB,EROTB,ERELSQ,H'
CXW...WRITE(6,*) HINTA,EROTA,HINTB,EROTB,ERELSQ,H
      NFINAL=0
      IF(lfault .ge. 4) THEN  
      CLOSE(6)
cx      CLOSE(11)
      GO TO 1111
      ENDIF
      linner=0
CXCCz
cx    write(6,*) 'SB after SELECT: ', SB

      CALL DVDQ

      CALL ENERGY
c     CALL GWRITE


CXCCa IF(NSELT.EQ.-2)CALL MPATH
      IF(NSELT.EQ.-2) THEN     
CXW...WRITE(6,*)  ' Reaction path calculation is not included '
      STOP 1    
      ENDIF
CXCCz
      I=0
CX   Normal mode analysis
CXP   IF(NSELT.EQ.-1)CALL NMODE(NATOMS,I)
      NC=6
      NX=NIP+6

      NXPLOT=NPLOT
CXP  402 GO TO 501
CXP   initiate integration with Runge-Kutta steps
  402 CONTINUE
       call runkut(NI,NID)
cx!!! next timestep starts here

  400 NC=NC+1
cXP   GO TO 600
CXP    progress one time step in the integration

      call admoul(NI,NID,ATIME)

  405 CONTINUE

CXCCC
cx    IF(NC .LE. NCROT+6) GO TO 8400 
 
 8000 CONTINUE
CXCCC
      IF(NFINAL.EQ.1)GO TO 414
      IF(NC.GE.NS)GO TO 450
      IF(NC.EQ.NX)GO TO 449
      IF(NSELT.EQ.1)GO TO 400
c*****************************************************
      CALL TEST
C
C        SUBROUTINE TEST CHECKS FOR EVENTS.
C             NTEST=0, A REACTION HAS NOT OCCURRED.
C  channel 1  NTEST=1, A + BC finished
C  channel 2  NTEST=2, AB + C finished
C  channel 3  NTEST=3, AC + B finished
C  dissoc.    NTEST=4, A + B + C finished
c       call gwrite 
c        write(*,9410) nc,(q(ik),ik=1,9)
 9410 format(i5,9f12.5)
cx6SEC test valid for TRIATomic systems
      IF(NTEST.NE.0) GO TO 410
cx    IF(NTEST.EQ.2.AND.NAST.EQ.1)GO TO 410
cx    IF(NTEST.EQ.2.AND.NAST.EQ.0)GO TO 410
cx    IF(NTEST.EQ.0.AND.NAST.NE.0)NAST=0
cx    IF(NTEST.EQ.1.AND.NAST.EQ.2)NAST=1
       call dvdq
       call energy
      GO TO 400
c*******************************************************
cx406 NAST=1
CXW...WRITE(6,15)
CXCCa.WRITE(6,910)NPATH
CXW...WRITE(6,910)NPATH,NC
cx    CALL DVDQ
CXCCz
cx    CALL ENERGY
cx    CALL GWRITE
cx    GO TO 400
  410 CONTINUE
cx6SEC  TRIATomic trajectory ended
cx  do not calculate average final rot. energy etc.
      CALL FINAL
      CALL PARTI
      CALL ENERGY
      CALL GFINAL

c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
c     isomm=somm
c     write(74,*) NTZ, NPATH, isomm !, (sommnum(ir),ir=1,somm)    
c     flush(74)
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


cx6SEC
c     write(66,*) 'npath', npath
      GO TO 451
CX 
  414 CONTINUE
      CALL DVDQ
      CALL ENERGY
      CALL FINAL
C
C        AVERAGE A AND B ROTATIONAL ENERGIES OVER NCROT CYCLES
C
      ERAVA(KRE)=EROTA
      ERAVB(KRE)=EROTB
      KRE=KRE+1
      IF(KRE.GT.NCROT)GO TO 411
      GO TO 400

  411 EROTA=0.0D0
      EROTB=0.0D0
      DO 412 I=1,NCROT
      EROTA=EROTA+ERAVA(I)
  412 EROTB=EROTB+ERAVB(I)
      EROTA=EROTA/FLOAT(NCROT)
      EROTB=EROTB/FLOAT(NCROT)
CXCCa
CXW...WRITE(6,*) ' aver. final EROTA,EROTB',EROTA,EROTB
CXCCz
      CALL DVDQ
      CALL ENERGY
      CALL GFINAL
CXCCa
      IF(LSEC .ge. 4) THEN
      ENMTB=EB(3) - EROTB
      TROTB=EROTB/C5*C1*2.d0/float(3-NLINB)
CX...WRITE(6,*)'  RSC: ENMTB,TROTB',ENMTB,TROTB
      IF(ENMTB .LT. 1.d0 .OR. EROTB .GT. 40.d0) THEN
cx!!! NTZ=NT+1 
      ENDIF
      ENDIF
CXCCz
      GO TO 451
CXCaa
 8400 CONTINUE
      NPATH=1
      NFINAL=1
      CALL DVDQ
      CALL ENERGY
      CALL FINAL
C         determination of initial rotational energy
C        AVERAGE A AND B ROTATIONAL ENERGIES OVER NCROT CYCLES
C
      ERAVA(KRE)=EROTA
      ERAVB(KRE)=EROTB
CXCCa
      aja(KRE)= sqrt(ama(1)**2 + ama(2)**2)
      ajb(KRE)= sqrt(amb(1)**2 + amb(2)**2)
CXCCz
CX   .WRITE(6,*) ':i',amb(3),EROTB,ajb(kre)
      KRE=KRE+1
      IF(KRE.GT.NCROT)GO TO 8411
      NFINAL=0
      GO TO 400
 8411 EROTA=0.0D0
      EROTB=0.0D0
CXCCa
      ajqai=0.d0          
      ajqbi=0.d0          
CX    AMBI(1)=ajb(1)
CXCCz
      DO 8412 I=1,NCROT
CXCCa
      ajqai=ajqai+aja(I)
      ajqbi=ajqbi+ajb(I)
CX    WRITE(6,*) ':i',ajqbi,EROTB,ajb(I)
CXCCz
      EROTA=EROTA+ERAVA(I)
 8412 EROTB=EROTB+ERAVB(I)
      ERTA0=EROTA/FLOAT(NCROT)
      ERT0=EROTB/FLOAT(NCROT)
CXCCa
      ajqai=ajqai/float(NCROT)
      ajqbi=ajqbi/float(NCROT)
CXCCz
CXW...WRITE(6,*) ' aver. init. EROTA,EROTB',ERTA0,ERT0 
      NFINAL=0
      KRE=1
      GO TO 8000
CXCzz
  447 DO 448 I=1,I3N
  448 P(I)=0.0
      NC=NC+6
      GO TO 402
  449 contINUE
      CALL DVDQ
      CALL ENERGY
      CALL GWRITE
      NX=NX+NIP
      IF(NSELT.EQ.1)GO TO 447
      GO TO 400
  450 CALL ENERGY
      CALL DVDQ
      CALL ENERGY
CXCCa    count trajectories failed to get to end
      iffa=iffa+1
      ifai(iffa)=ntz
cx    NTZ=NTZ-1
      IF(iffa .GT. 500) THEN
      WRITE(9,*) 'failed trajs',ifai
      GO TO 1111
      ENDIF
CXCCz
      CALL DVDQ
      CALL ENERGY
      CALL GWRITE
      GO TO 451
 1111 CONTINUE
CXUP   WRITE(9,1905) EVIB0P(1),ERTB0P(1),EVIBP(1),EROTBP(1),
CXUP * SER1P(1),ERELSP(1),SBP(1),ANG4P(1),
CXUP * AMBIP(1),AMBFP(1),LINNERP(1),KSEED(mype+1)
CXUP    do i=2,NT
CXUP    WRITE(9,1905) EVIB0P(i),ERTB0P(i),EVIBP(i),EROTBP(i),
CXUP *  SER1P(i),ERELSP(i),SBP(i),ANG4P(i),
CXUP *  AMBIP(i),AMBFP(i),LINNERP(i)
CXUP    end do
CXP     print data here instead of in GFINAL
CXP      from the files in COMMON/parall/
CXUP  if(mype.ne.0) then
CXP    get cpu time (real time at the end of the run on PE 0)
CXUP  time1=rtc()/1.5e8
CXP     time used by PE=mype
CXUP  timpa=time1-time0
CXP    get cpu time (real time at the end of the run on PE 0)
CXUP  time2=rtc()/1.5e8
 1905 FORMAT(10f20.7,2I8)
      GO TO 1110
      END
C
