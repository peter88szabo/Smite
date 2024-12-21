      subroutine initialize_potential()
        implicit double precision (a-h,o-z)
	DOUBLE PRECISION :: r, energy, dvdr
        DOUBLE PRECISION CART,ANUZERO
        INTEGER NULBL,NFLAG,NASURF,NDER
        INTEGER N3ATOM, NATOM, ISURF, JSURF
!
        PARAMETER (NATOM=25)
        PARAMETER (ISURF = 5)
        PARAMETER (JSURF = ISURF*(ISURF+1)/2)

        COMMON/USROCM/ PENGYGS,PENGYES(ISURF), 
     +               PENGYIJ(JSURF), 
     +               DGSCART(NATOM,3),DESCART(NATOM,3,ISURF), 
     +                DIJCART(NATOM,3,JSURF)

        COMMON/USRICM/ CART(NATOM,3),ANUZERO, 
     +               NULBL(NATOM),NFLAG(20),  
     +               NASURF(ISURF+1,ISURF+1),NDER


        INTEGER :: init
        data init /0/
        save init
        natoms = 3
        nder =  1
        nasurf =  0
	nasurf(1,1) = 1
        nflag(1) = 1 
        nflag(2) = 1
        nflag(3) = 0
        nflag(4) = 0
        nflag(5) = 0 

        init = 0
        call prepot
        init = 1
      
      end subroutine initialize_potential

      subroutine get_potential(q,Natoms,Nbeads,V,dVdq,info)
        implicit double precision (a-h,o-z)
        INTEGER :: Natoms, Nbeads
        DOUBLE PRECISION :: q(3,Natoms,Nbeads)
        DOUBLE PRECISION :: dVdq(3,Natoms,Nbeads), V(Nbeads)
        INTEGER :: k
	DOUBLE PRECISION :: r, energy, dvdr
        DOUBLE PRECISION CART,ANUZERO
        INTEGER NULBL,NFLAG,NASURF,NDER
        INTEGER N3ATOM, NATOM, ISURF, JSURF
	INTEGER :: INFO
!
        PARAMETER (NATOM=25)
        PARAMETER (ISURF = 5)
        PARAMETER (JSURF = ISURF*(ISURF+1)/2)

        COMMON/USROCM/ PENGYGS,PENGYES(ISURF), 
     +               PENGYIJ(JSURF), 
     +               DGSCART(NATOM,3),DESCART(NATOM,3,ISURF), 
     +                DIJCART(NATOM,3,JSURF)

        COMMON/USRICM/ CART(NATOM,3),ANUZERO, 
     +               NULBL(NATOM),NFLAG(20),  
     +               NASURF(ISURF+1,ISURF+1),NDER


        INTEGER :: init
        data init /0/
        save init
        natoms = 3
        nder =  1
        nasurf =  0
	nasurf(1,1) = 1
	info = 0 
        nflag(1) = 1 
        nflag(2) = 1
        nflag(3) = 0
        nflag(4) = 0
        nflag(5) = 0 
   
     
        init = 1
        v = 0.d0 
        dvdq(:,:,:)  = 0.d0
        dvdr = 0.d0 
        cart(:,:) = 0.d0
        do k = 1, Nbeads
            call qtocart(q,Natoms,natom,Nbeads,k,cart)
            call pot
            
            V(k) = PENGYGS
	do i = 1,3
	do j = 1,natoms
	dvdq(i,j,k) = dgscart(j,i) 
	end do 
	end do 
        END DO
      end subroutine get_potential

      subroutine qtocart(q0,nat,natom,nb,inb,cart)      
        implicit none
        DOUBLE PRECISION :: q0(3,nat,nb), cart(natom,3)
        INTEGER :: i,j,k,inb
        INTEGER :: nat, natom,nb 
	do i= 1,3
	do j = 1,nat
	cart(j,i) = q0(i,j,inb) 
	end do 
	end do 
    !
        return
      end subroutine qtocart

C                                                                               
      SUBROUTINE PREPOT                                                         
C                                                                               
C   System:          FH2                                                        
C   Functional form: eLEPS (extended-London-Eyring-Polanyi-Sato)                
C                    plus E3C (a three-center term).                            
C   Common name:     no. 5Z                                                     
C   Reference:       unpublished                                                
C                    (see Steckler's comp index)                                
C                                                                               
C   Notes:           In this version of the potential the AB and AC             
C                    Sato parameters are functions of PHI, where PHI is         
C                    the angle of A-to-BC vector with the BC axis.              
C                    The BC Sato parameter is a function of BC bond             
C                    length and the angle PHI.                                  
C                                                                               
C   PREPOT must be called once before any calls to POT.                         
C   The potential parameters are included in DATA statements                    
C   and in the block data subprogram PTPACM.                                    
C   Coordinates, potential energy, and derivatives are passed                   
C   The potential energy in the three asymptotic valleys are                    
C   stored in the common block ASYCM:                                           
C                  COMMON /ASYCM/ EASYAB, EASYBC, EASYAC                        
C   The potential energy in the AB valley, EASYAB, is equal to the potential    
C   energy of the H "infinitely" far from the FH diatomic, with the             
C   FH diatomic at its equilibrium configuration.  Similarly, the terms         
C   EASYBC and EASYAC represent the H2 and the FH asymptotic valleys,           
C   respectively.                                                               
C   All the information passed through the common blocks PT1CM and ASYCM        
C   is in Hartree atomic units.                                                 
C                                                                               
C   This potential is written such that:                                        
C                  R(1) = R(F-H)                                                
C                  R(2) = R(H-H)                                                
C                  R(3) = R(H-F)                                                
C   The zero of energy is at F "infinitely" far from the H2 diatomic.           
C                                                                               
C   The flags that indicate what calculations should be carried out in          
C   the potential routine are passed through the common block PT2CM:            
C   where:                                                                      
C        NASURF - which electronic states availalble                            
C                 (1,1) = 1 as only gs state available                          
C        NDER  = 0 => no derivatives should be calculated                       
C        NDER  = 1 => calculate first derivatives                               
C        NFLAG - these integer values can be used to flag options               
C                within the potential;                                          
C                                                                               
C                                                                               
C   Potential parameters' default settings                                      
C                  Variable            Default value                            
C                  NDER                1                                        
C                  NFLAG(18)           6                                        
C                                                                               
         IMPLICIT DOUBLE PRECISION (A-H,O-Z)                                    
C                                                                               
      CHARACTER*75 REF(5)                                                       
C                                                                               
      PARAMETER (N3ATOM = 75)                                                   
      PARAMETER (ISURF = 5)                                                     
      PARAMETER (JSURF = ISURF*(ISURF+1)/2)                             
      PARAMETER (PI = 3.141592653589793D0)                                      
      PARAMETER (NATOM = 25)                                                    
C                                                                               
      COMMON /PT3CM/ EZERO(ISURF+1)                                             
C                                                                               
      COMMON/INFOCM/ CARTNU(NATOM,3),INDEXES(NATOM),                            
     +               IRCTNT,NATOMS,ICARTR,MDER,MSURF,REF                        
C                                                                               
C                                                                               
      COMMON/USRICM/ CART(NATOM,3),ANUZERO,                                     
     +               NULBL(NATOM),NFLAG(20),                                    
     +               NASURF(ISURF+1,ISURF+1),NDER                               
C                                                                               
      COMMON /ASYCM/ EASYAB,EASYBC,EASYAC                                       
C                                                                               
         PARAMETER (CKCAU = 627.5095D0)                                         
         PARAMETER (CANGAU = 0.52917706D0)                                      
C                                                                               
         COMMON /SATOCM/ DM(3), RE(3), BETA(3)                                  
C                                                                               
C      LOGICAL LCOL, LDERIV, LZERO                                              
         COMMON /PRECM/  Z(3),Q(3),XJ(3),DQ(3,3),DJ(3,3),                       
     +                   DZ(3,3),RR(3),A1,A2,A,B,C,D,F,G,                       
     +                   BET,BETP,ZFH0,ZHH0,ZHHST,ZA,ZB,                        
     +                   DEG1,DEG2,DEG3,DEG4,DEG3P,C1,C2,                       
     +                   C3,C4,C5,C6,C3C,ALF,ALFP,ALFT,                         
     +                   P3C,Q3C,ZHHDIF, OMP3C,OMQ3C,HA2,G2                     
C                                                                               
      IF(NATOMS.GT.25) THEN                                                     
         WRITE(NFLAG(18),1111)                                                  
 1111    FORMAT(2X,'STOP. NUMBER OF ATOMS EXCEEDS ARRAY DIMENSIONS')            
         STOP                                                                   
      END IF                                                                    
C                                                                               
         WRITE (NFLAG(18), 600) DM, RE, BETA                                    
         WRITE (NFLAG(18), 610) A, ZFH0, A1, ZA, A2, ZB, B, C1, C,              
     *                      C2, D, C3, F, C4, G, C5, BET, C6, BETP              
         WRITE (NFLAG(18), 620) C3C, ALF, ALFP, ALFT, P3C, Q3C                  
C                                                                               
600   FORMAT (/,2X,T5,'PREPOT has been called for the FH2 ',                    
     *                'extended-LEPS plus three-center term ',                  
     *        /,2X,T5,'potential energy surface no. 5Z',                        
     *       //,2X,T5,'Potential energy surface parameters:',                   
     *        /,2X,T5,'Bond', T47, 'F-H', T58, 'H-H', T69, 'H-F',               
     *        /,2X,T5,'Dissociation energies (kcal/mol):',                      
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,2X,T5,'Equilibrium bond lengths (Angstroms):',                  
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,2X,T5,'Morse beta parameters (Angstroms**-1):',                 
     *        T44, F10.5, T55, F10.5, T66, F10.5)                               
610   FORMAT (/,2X,T5,'HH Sato parameter fit',                                  
     *             T40,'FH Sato parameter fit',                                 
     * /,2X,T5,'A',   T10,'=',T12,1PE13.5, T40,'ZFH0',T46,'=',T48,E13.5,        
     * /,2X,T5,'A1',  T10,'=',T12,E13.5, T40,'ZA', T46,'=',T48,1PE13.5,         
     * /,2X,T5,'A2',  T10,'=',T12,E13.5, T40,'ZB', T46,'=',T48,E13.5,           
     * /,2X,T5,'B',   T10,'=',T12,E13.5, T40,'C1', T46,'=',T48,E13.5,           
     * /,2X,T5,'C',   T10,'=',T12,E13.5, T40,'C2', T46,'=',T48,E13.5,           
     * /,2X,T5,'D',   T10,'=',T12,E13.5, T40,'C3',T46,'=',T48,E13.5,            
     * /,2X,T5,'F',   T10,'=',T12,E13.5, T40,'C4',   T46,'=',T48,E13.5,         
     * /,2X,T5,'G',   T10,'=',T12,E13.5, T40,'C5',   T46,'=',T48,E13.5,         
     * /,2X,T5,'BET', T10,'=',T12,E13.5, T40,'C6',   T46,'=',T48,E13.5,         
     * /,2X,T5,'BETP',T10,'=',T12,E13.5)                                        
620   FORMAT(/, 2X, T5, 'Parameters for the three-center term:',                
     * /,2X,T5,'C3C', T10,'=',T12,E13.5,T40,'ALF', T46,'=',T48,E13.5,           
     * /,2X,T5,'ALFP',T10,'=',T12,E13.5,T40,'ALFT',T46,'=',T48,E13.5,           
     * /,2X,T5,'P3C', T10,'=',T12,E13.5,T40,'Q3C', T46,'=',T48,E13.5)           
C                                                                               
      DO 10 I = 1,3                                                             
         DM(I) = DM(I) / CKCAU                                                  
         RE(I) = RE(I) / CANGAU                                                 
         BETA(I) = BETA(I) * CANGAU                                             
   10 CONTINUE                                                                  
C                                                                               
C   Set the energy of the potential in the each of the three asymptotic regions 
C                                                                               
         EASYAB = DM(1)                                                         
         EASYBC = DM(2)                                                         
         EASYAC = DM(3)                                                         
C                                                                               
C   USEFUL CONSTANTS                                                            
C                                                                               
      ZHHDIF = ZHHST - ZHH0                                                     
      OMP3C = 1.D0 - P3C                                                        
      OMQ3C = 1.D0 - Q3C                                                        
      HA2 = 0.5D0 * A2                                                          
      G2 = G * G                                                                
C                                                                               
C                                                                               
      EZERO(1)=DM(2)                                                            
C                                                                               
       DO I=1,5                                                                 
          REF(I) = ' '                                                          
       END DO                                                                   
C                                                                               
       REF(1)='Unpublished'                                                     
C                                                                               
      INDEXES(1) = 9                                                            
      INDEXES(2) = 1                                                            
      INDEXES(3) = 1                                                            
C                                                                               
C                                                                               
C                                                                               
      IRCTNT=2                                                                  
C                                                                               
      CALL POTINFO                                                              
C                                                                               
      CALL ANCVRT                                                               
C                                                                               
      RETURN                                                                    
      END                                                                       
C                                                                               
      SUBROUTINE POT                                                            
C                                                                               
C   The potential energy in the AB valley, EASYAB, is equal to the potential    
C   energy of the H "infinitely" far from the FH diatomic, with the             
C   FH diatomic at its equilibrium configuration.  Similarly, the terms         
C   EASYBC and EASYAC represent the H2 and the FH asymptotic valleys,           
C   respectively.                                                               
C                                                                               
C   This potential is written such that:                                        
C                  R(1) = R(F-H)                                                
C                  R(2) = R(H-H)                                                
C                  R(3) = R(H-F)                                                
C   The zero of energy is at F "infinitely" far from the H2 diatomic.           
C                                                                               
C      ENTRY POT                                                                
         IMPLICIT DOUBLE PRECISION (A-H,O-Z)                                    
C                                                                               
      CHARACTER*75 REF(5)                                                       
C                                                                               
      PARAMETER (N3ATOM = 75)                                                   
      PARAMETER (ISURF = 5)                                                     
      PARAMETER (JSURF = ISURF*(ISURF+1)/2)                             
      PARAMETER (PI = 3.141592653589793D0)                                      
      PARAMETER (NATOM = 25)                                                    
C                                                                               
      COMMON /PT1CM/ R(N3ATOM),ENGYGS,DEGSDR(N3ATOM)                            
      COMMON /PT3CM/ EZERO(ISURF+1)                                             
      COMMON /PT4CM/ ENGYES(ISURF),DEESDR(N3ATOM,ISURF)                         
      COMMON /PT5CM/ ENGYIJ(JSURF),DEIJDR(N3ATOM,JSURF)                         
C                                                                               
      COMMON/INFOCM/ CARTNU(NATOM,3),INDEXES(NATOM),                            
     +               IRCTNT,NATOMS,ICARTR,MDER,MSURF,REF                        
C                                                                               
      COMMON/USROCM/ PENGYGS,PENGYES(ISURF),                                    
     +               PENGYIJ(JSURF),                                            
     +               DGSCART(NATOM,3),DESCART(NATOM,3,ISURF),                   
     +               DIJCART(NATOM,3,JSURF)                                     
C                                                                               
      COMMON/USRICM/ CART(NATOM,3),ANUZERO,                                     
     +               NULBL(NATOM),NFLAG(20),                                    
     +               NASURF(ISURF+1,ISURF+1),NDER                               
C                                                                               
      COMMON /ASYCM/ EASYAB,EASYBC,EASYAC                                       
C                                                                               
         PARAMETER (CKCAU = 627.5095D0)                                         
         PARAMETER (CANGAU = 0.52917706D0)                                      
         PARAMETER (TDEG  = 57.29577951D0)                                      
         PARAMETER (EXLARG = 80.D0)                                             
C                                                                               
C   EXLARG IS LN(LARGEST FLOATING POINT NUMBER ALLOWED)                         
C                                                                               
         COMMON /SATOCM/ DM(3), RE(3), BETA(3)                                  
C                                                                               
         COMMON /PRECM/  Z(3),Q(3),XJ(3),DQ(3,3),DJ(3,3),                       
     +                   DZ(3,3),RR(3),A1,A2,A,B,C,D,F,G,                       
     +                   BET,BETP,ZFH0,ZHH0,ZHHST,ZA,ZB,                        
     +                   DEG1,DEG2,DEG3,DEG4,DEG3P,C1,C2,                       
     +                   C3,C4,C5,C6,C3C,ALF,ALFP,ALFT,                         
     +                   P3C,Q3C,ZHHDIF, OMP3C,OMQ3C,HA2,G2                     
C                                                                               
      LOGICAL LCOL, LDERIV, LZERO                                               
C                                                                               
      DATA LDERIV /.FALSE./                                                     
C                                                                               
      CALL CARTOU                                                               
      CALL CARTTOR                                                              
C                                                                               
C   Check the values of NASURF and NDER for validity.                           
C                                                                               
      IF (NASURF(1,1) .EQ. 0) THEN                                              
         WRITE(NFLAG(18), 900) NASURF(1,1)                                      
         STOP                                                                   
      ENDIF                                                                     
C                                                                               
C   Check the value of NDER                                                     
C                                                                               
         IF (NDER .GT. 1) THEN                                                  
             WRITE (NFLAG(18), 910) NDER                                        
             STOP 'POT 2'                                                       
         ENDIF                                                                  
C                                                                               
C   The logical LDERIV is used to flag the derivative calculations              
C   in the code; LDERIV is set equal to .TRUE. if NDER = 1.                     
C                                                                               
         IF (NDER .EQ. 1) LDERIV = .TRUE.                                       
C                                                                               
C   TEST R'S FOR PHYSICAL VALUES                                                
C                                                                               
      DO 20 I = 1,3                                                             
         RR(I) = R(I)                                                           
   20 CONTINUE                                                                  
      DO 30 I1 = 1,3                                                            
         I2 = MOD(I1,3) + 1                                                     
         I3 = MOD(I2,3) + 1                                                     
         T = RR(I2) + RR(I3)                                                    
         IF (RR(I1) .GT. T) RR(I1) = T                                          
   30 CONTINUE                                                                  
      R12 = RR(1) * RR(1)                                                       
      R22 = RR(2) * RR(2)                                                       
      R32 = RR(3) * RR(3)                                                       
C                                                                               
C   CAPR IS THE F TO H2 DISTANCE                                                
C                                                                               
      CAPR2 = 0.5D0 * (R12 + R32 - 0.5D0 * R22)                                 
      IF (CAPR2 .LT. 0.D0) CAPR2 = 0.0D0                                        
      CAPR = SQRT(CAPR2)                                                        
C                                                                               
C   PHI IS THE ANGLE BETWEEN CAPR AND RHH                                       
C   PHI IS RESTRICTED TO BE BETWEEN O AND 90 DEGREES                            
C                                                                               
      T = 0.5D0 * (R32 - R12)                                                   
      IF (ABS(T) .GT. 1.D-7) T = T / (RR(2) * CAPR)                             
      T2 = 1.0D0                                                                
      SGNPHI = SIGN(T2,T)                                                       
      COSPHI = SGNPHI * T                                                       
      COSPHI = MIN(COSPHI,T2)                                                   
      PHI = ACOS(COSPHI)                                                        
      SINP2 = 1.D0 - COSPHI * COSPHI                                            
      IF(SINP2 .LT. 0.D0) SINP2 = 0.D0                                          
      SINPHI = SQRT(SINP2)                                                      
C                                                                               
C   LCOL IS SET TRUE IF COLLINEAR GEOMETRY                                      
C                                                                               
      LCOL = ABS(SINPHI) .LT. 1.D-6 .OR. ABS(CAPR) .LT. 1.D-6                   
C                                                                               
C   COMPUTE COLLINEAR SATO PARAMETERS                                           
C                                                                               
      Z(1) = ZFH0                                                               
      T = A * (RR(2) - B)                                                       
      IF (.NOT.(T .LT. EXLARG)) GO TO 35                                        
         T = EXP(T)                                                             
         T1 = 1.D0 / T                                                          
         HSECH = 1.D0/(T + T1)                                                  
         ZHH = A1 + HA2 * ((T - T1) * HSECH + 1.D0)                             
         GO TO 36                                                               
   35 CONTINUE                                                                  
         HSECH = 0.D0                                                           
         ZHH = A1 + A2                                                          
   36 CONTINUE                                                                  
      T = RR(2) - F                                                             
      EXX = C * EXP(D * T * T)                                                  
      ZHH = ZHH - EXX                                                           
      Z(2) = ZHH                                                                
      IF (.NOT.LDERIV) GO TO 50                                                 
         DO 40 I = 1,3                                                          
            DO 40 J = 1,3                                                       
               DZ(I,J) = 0.D0                                                   
 40   CONTINUE                                                                  
C                                                                               
C  ONLY THE HH SATO PARAMETER HAS A NONZERO DERIVATIVE - W.R.T. R2              
C                                                                               
         DZ(2,2) = 2.D0 * A * A2 * HSECH * HSECH                                
     *      - 2.D0 * D * T * EXX                                                
   50 CONTINUE                                                                  
C                                                                               
C   COMPUTE NONCOLLINEAR CONTRIBUTIONS TO SATO PARAMETERS                       
C                                                                               
      IF (LCOL) GO TO 160                                                       
C                                                                               
C   COMPUTE Z(PHI) FOR FH SATO PARAMETER                                        
C   DZPHI IS THE DERIVATIVE OF Z(PHI) W.R.T. PHI                                
C                                                                               
         ZPHI = ZA*SINP2 + ZB*SINP2*SINP2                                       
         TEM = 2.0D0*SINPHI*COSPHI                                              
         DZPHI = (ZA+2.0D0*ZB*SINP2)*TEM                                        
C                                                                               
C   COMPUTE SCALE FACTOR TO DAMP OUT CORRECTION AT LARGE DISTANCE               
C                                                                               
         SCALE1 = CAPR2 / (G2 + CAPR2)                                          
         Z(1) = Z(1) + SCALE1 * ZPHI                                            
C                                                                               
C   COMPUTE SCALE FACTOR FOR HH SATO PARAMETER                                  
C                                                                               
         SCALE2 = (ZHH - ZHH0) / ZHHDIF                                         
C                                                                               
C   COMPUTE PHI DEPENDENT PART OF HH SATO PARAMETER                             
C                                                                               
         CPHI = (BET + BETP * SINP2) * SINP2                                    
C                                                                               
C   COMPUTE HH SATO PARAMTER                                                    
C                                                                               
         Z(2) = Z(2) + SCALE1 * SCALE2 * CPHI                                   
         IF (.NOT.LDERIV) GO TO 150                                             
C                                                                               
C   FIRST, DERIVATIVES OF PHI W.R.T. R1, R2, R3                                 
C                                                                               
            T = 1.D0 / CAPR                                                     
            T1 = 0.5D0 * COSPHI * T                                             
            T2 = SGNPHI / RR(2)                                                 
            DPHI1 = RR(1) * (T1 + T2)                                           
            DPHI3 = RR(3) * (T1 - T2)                                           
            T1 = 1.D0 / SINPHI                                                  
            T2 = T1 * T                                                         
            DPHI1 = DPHI1 * T2                                                  
            DPHI3 = DPHI3 * T2                                                  
            DPHI2 = RR(2) * COSPHI * T1 * (1.D0 / R22 - 0.25D0 / CAPR2)         
            T1 = (1.D0 - SCALE1) / CAPR2                                        
            T2 = T1 * ZPHI                                                      
C                                                                               
C   NEXT, DERIVATIVES OF FH SATO PARAMETERS                                     
C                                                                               
            DZ(1,1) = SCALE1 * (T2 * RR(1) + DZPHI * DPHI1)                     
            DZ(1,2) = SCALE1 * (-0.5D0 * T2 * RR(2) + DZPHI * DPHI2)            
            DZ(1,3) = SCALE1 * (T2 * RR(3) + DZPHI * DPHI3)                     
            T1 = T1 * CPHI                                                      
C                                                                               
C   DERIVATIVE OF THE PHI DEPENDENT PART OF HH SATO W.R.T. PHI                  
C                                                                               
            DCPHI = 2.D0 * (BET + 2.D0 * BETP * SINP2) * SINPHI * COSPHI        
            T2 = SCALE1 * SCALE2                                                
C                                                                               
C   LAST, DERIVATIVES OF HH SATO PARAMETER                                      
C                                                                               
            DZ(2,1) = T2 * (T1 * RR(1) + DCPHI * DPHI1)                         
            DZ(2,2) = DZ(2,2) * (1.D0 + SCALE1 * CPHI / ZHHDIF)                 
     *         + T2 * (-0.5D0 * T1 * RR(2) + DCPHI * DPHI2)                     
            DZ(2,3) = T2 * (T1 * RR(3) + DCPHI * DPHI3)                         
  150    CONTINUE                                                               
  160 CONTINUE                                                                  
C                                                                               
C   COMPUTE POTENTIAL FROM LEPS FORM                                            
C                                                                               
      ENGYGS = 0.D0                                                             
      Z(3) = Z(1)                                                               
      IF (.NOT.LDERIV) GO TO 170                                                
         DO 165 I = 1,3                                                         
            DZ(3,I) = DZ(1,I)                                                   
            DEGSDR(I) = 0.D0                                                    
            Q(I) = 0.D0                                                         
            XJ(I) = 0.D0                                                        
            DO 162 J = 1,3                                                      
               DQ(I,J) = 0.D0                                                   
               DJ(I,J) = 0.D0                                                   
  162       CONTINUE                                                            
  165    CONTINUE                                                               
  170 CONTINUE                                                                  
      LZERO = .TRUE.                                                            
      DO 200 I = 1,3                                                            
         EX = EXP(BETA(I) * (RE(I) - RR(I)))                                    
         LZERO = LZERO .AND. EX.EQ.0.0D0                                        
         IF (EX.EQ.0.0D0) GO TO 195                                             
            ZP1 = Z(I) + 1.D0                                                   
            ZP3 = ZP1 + 2.D0                                                    
            OP3Z = 1.D0 + 3.D0 * Z(I)                                           
            T = 0.25D0 * DM(I) * EX / ZP1                                       
            T1 = ZP3 * EX                                                       
            T2 = OP3Z * EX                                                      
C                                                                               
C   Q AND XJ ARE THE COULUMB AND EXCHANGE PARTS, INDEX I RUNS OVER THE          
C   COORDINATES OF WHICH THEY ARE EXPLICIT FUNCTIONS                            
C                                                                               
            Q(I) = T * (T1 - 2.D0  * OP3Z)                                      
            XJ(I) = T * (T2 - 2.D0  * ZP3)                                      
C                                                                               
C   PUT SUM OF Q'S IN ENERGY                                                    
C                                                                               
            ENGYGS = ENGYGS + Q(I)                                              
            IF (.NOT.LDERIV) GO TO 190                                          
C                                                                               
C   DERIVATIVE OF Q AND ZJ W.R.T. R1, R2, R3                                    
C   ELEMENT I,J IS THE DERIVATIVE OF Q(I) W.R.T. RJ                             
C                                                                               
               T3 = 2.D0 * T * (EX + 2.D0) / ZP1                                
               DO 180 J = 1,3                                                   
                  DQ(I,J) = -T3 * DZ(I,J)                                       
                  DJ(I,J) = -DQ(I,J)                                            
  180          CONTINUE                                                         
               T = 2.D0*T*BETA(I)                                               
               DQ(I,I) = DQ(I,I) - T * (T1 - OP3Z)                              
               DJ(I,I) = DJ(I,I) - T * (T2 - ZP3)                               
  190       CONTINUE                                                            
  195    CONTINUE                                                               
  200 CONTINUE                                                                  
      IF (LZERO) GO TO 270                                                      
         XJX = 0.D0                                                             
C                                                                               
C   COMPUTE THE TOTAL EXCHANGE TERM                                             
C                                                                               
         DO 230 I1 = 1,3                                                        
            I2 = MOD(I1,3) + 1                                                  
            T = XJ(I1) - XJ(I2)                                                 
            XJX = XJX + T * T                                                   
            IF (.NOT.LDERIV) GO TO 220                                          
C                                                                               
C   PUT THE DERIVATIVES OF THE TOTAL EXCHANGE TERM IN DEGSDR                    
C                                                                               
               DO 210 J = 1,3                                                   
                  DEGSDR(J) = DEGSDR(J) + T * (DJ(I1,J) - DJ(I2,J))             
  210          CONTINUE                                                         
  220       CONTINUE                                                            
  230    CONTINUE                                                               
         XJX = SQRT(0.5D0 * XJX)                                                
C                                                                               
C   COMPUTE THE LEPS PART OF THE POTENTIAL ENERGY                               
C                                                                               
         ENGYGS = ENGYGS - XJX                                                  
         IF (.NOT.LDERIV) GO TO 260                                             
C                                                                               
C   COMPUTE DERIVATIVES OF THE LEPS PART                                        
C                                                                               
            IF(ABS(XJX) .GT. 1.D-14) T = 0.5D0 / XJX                            
            DO 250 J = 1,3                                                      
C                                                                               
C   THE DERIVATIVE OF THE EXCHANGE PART MUST BE DIVIDED BY 2*XJX                
C                                                                               
               DEGSDR(J) = -T * DEGSDR(J)                                       
               DO 240  I = 1,3                                                  
C                                                                               
C   THEN ADD IN THE DERIVATIVE OF Q                                             
C                                                                               
                  DEGSDR(J) = DEGSDR(J) + DQ(I,J)                               
  240          CONTINUE                                                         
  250       CONTINUE                                                            
  260    CONTINUE                                                               
  270 CONTINUE                                                                  
C                                                                               
C   EZERO ADDED TO PUT ZERO AT REQ FOR REACTANTS                                
C                                                                               
      ENGYGS = ENGYGS + EZERO(1)                                                
C                                                                               
C   COMPUTE THE 3-CENTER TERM                                                   
C                                                                               
      RSUM = RR(1) + RR(3)                                                      
      RDIF = RR(1) - RR(3)                                                      
      RDIF2 = RDIF * RDIF                                                       
      T1 = -ALFP * RSUM                                                         
      EX1 = C3C * EXP(T1 * RSUM)                                                
      T2 = -ALF * RDIF                                                          
      EX2 = OMP3C * EXP(T2 * RDIF)                                              
      T3 = -ALFT * RDIF2                                                        
      EX3 = P3C * EXP(T3 * RDIF2)                                               
      R1I = 1.D0 / RR(1)                                                        
      R3I = 1.D0 / RR(3)                                                        
      T4 = R1I*R3I                                                              
C                                                                               
C   COSTH IS THE ANGLE BETWEEN R1 AND R3                                        
C                                                                               
      COSTH = 0.5D0 * (R12 + R32 - R22) * T4                                    
      T5 = OMQ3C * COSTH                                                        
      T = Q3C + T5 * COSTH                                                      
C                                                                               
C   E3C CONTAINS THE THREE CENTER CORRECTION TO THE ENERGY                      
C                                                                               
      E3C = (EX2 + EX3) * T * EX1                                               
      ENGYGS = ENGYGS + E3C                                                     
      IF (.NOT.LDERIV) GO TO 280                                                
C                                                                               
C   COMPUTE DERIVATIVE OF THE 3C TERM                                           
C   FIRST, DERIVATIVE OF COSTH                                                  
C                                                                               
         DCOS1 = R3I - COSTH*R1I                                                
         DCOS2 = - RR(2)*T4                                                     
         DCOS3 = R1I - COSTH*R3I                                                
         T4 = EX2 + EX3                                                         
         T5 = T4 * T5                                                           
C                                                                               
C   NOW, DERIVATIVES OF E3C                                                     
C                                                                               
         DE3C1 = 2.D0*((T2 * EX2 + 2.D0 * T3 * RDIF * EX3 + T1 * T4)            
     *      * T + T5 * DCOS1) * EX1                                             
         DE3C2 = 2.D0 * T5 * DCOS2 * EX1                                        
         DE3C3 = 2.D0*((-T2 * EX2 - 2.D0 * T3 * RDIF * EX3 + T1 * T4)           
     *      * T + T5 * DCOS3) * EX1                                             
         DEGSDR(1) = DEGSDR(1) + DE3C1                                          
         DEGSDR(2) = DEGSDR(2) + DE3C2                                          
         DEGSDR(3) = DEGSDR(3) + DE3C3                                          
  280 CONTINUE                                                                  
600   FORMAT (/,2X,T5,'PREPOT has been called for the FH2 ',                    
     *                'extended-LEPS plus three-center term ',                  
     *        /,2X,T5,'potential energy surface no. 5Z',                        
     *       //,2X,T5,'Potential energy surface parameters:',                   
     *        /,2X,T5,'Bond', T47, 'F-H', T58, 'H-H', T69, 'H-F',               
     *        /,2X,T5,'Dissociation energies (kcal/mol):',                      
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,2X,T5,'Equilibrium bond lengths (Angstroms):',                  
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,2X,T5,'Morse beta parameters (Angstroms**-1):',                 
     *        T44, F10.5, T55, F10.5, T66, F10.5)                               
610   FORMAT (/,2X,T5,'HH Sato parameter fit',                                  
     *             T40,'FH Sato parameter fit',                                 
     * /,2X,T5,'A',   T10,'=',T12,1PE13.5, T40,'ZFH0',T46,'=',T48,E13.5,        
     * /,2X,T5,'A1',  T10,'=',T12,E13.5, T40,'ZA', T46,'=',T48,1PE13.5,         
     * /,2X,T5,'A2',  T10,'=',T12,E13.5, T40,'ZB', T46,'=',T48,E13.5,           
     * /,2X,T5,'B',   T10,'=',T12,E13.5, T40,'C1', T46,'=',T48,E13.5,           
     * /,2X,T5,'C',   T10,'=',T12,E13.5, T40,'C2', T46,'=',T48,E13.5,           
     * /,2X,T5,'D',   T10,'=',T12,E13.5, T40,'C3',T46,'=',T48,E13.5,            
     * /,2X,T5,'F',   T10,'=',T12,E13.5, T40,'C4',   T46,'=',T48,E13.5,         
     * /,2X,T5,'G',   T10,'=',T12,E13.5, T40,'C5',   T46,'=',T48,E13.5,         
     * /,2X,T5,'BET', T10,'=',T12,E13.5, T40,'C6',   T46,'=',T48,E13.5,         
     * /,2X,T5,'BETP',T10,'=',T12,E13.5)                                        
620   FORMAT(/, 2X, T5, 'Parameters for the three-center term:',                
     * /,2X,T5,'C3C', T10,'=',T12,E13.5,T40,'ALF', T46,'=',T48,E13.5,           
     * /,2X,T5,'ALFP',T10,'=',T12,E13.5,T40,'ALFT',T46,'=',T48,E13.5,           
     * /,2X,T5,'P3C', T10,'=',T12,E13.5,T40,'Q3C', T46,'=',T48,E13.5)           
 900  FORMAT(/,2X,T5,13HNASURF(1,1) =,I5,                                       
     *       /,2X,T5,24HThis value is unallowed.                                
     *       /,2X,T5,31HOnly gs surface=>NASURF(1,1)=1 )                        
910   FORMAT(/, 2X,'POT has been called with NDER = ',I5,                       
     *       /, 2X, 'This value of NDER is not allowed in this ',               
     *              'version of the potential.')                                
C                                                                               
      CALL EUNITZERO                                                            
      IF(NDER.NE.0) THEN                                                        
         CALL RTOCART                                                           
         CALL DEDCOU                                                            
      ENDIF                                                                     
C                                                                               
      RETURN                                                                    
      END                                                                       
C                                                                               
C*****                                                                          
C                                                                               
         BLOCK DATA PTPACM                                                      
         IMPLICIT DOUBLE PRECISION (A-H,O-Z)                                    
C                                                                               
      CHARACTER*75 REF(5)                                                       
C                                                                               
      PARAMETER (N3ATOM = 75)                                                   
      PARAMETER (ISURF = 5)                                                     
      PARAMETER (JSURF = ISURF*(ISURF+1)/2)                             
      PARAMETER (PI = 3.141592653589793D0)                                      
      PARAMETER (NATOM = 25)                                                    
C                                                                               
      COMMON /PT3CM/ EZERO(ISURF+1)                                             
C                                                                               
      COMMON/INFOCM/ CARTNU(NATOM,3),INDEXES(NATOM),                            
     +               IRCTNT,NATOMS,ICARTR,MDER,MSURF,REF                        
C                                                                               
C                                                                               
      COMMON/USRICM/ CART(NATOM,3),ANUZERO,                                     
     +               NULBL(NATOM),NFLAG(20),                                    
     +               NASURF(ISURF+1,ISURF+1),NDER                               
C                                                                               
      COMMON /ASYCM/ EASYAB,EASYBC,EASYAC                                       
C                                                                               
         COMMON /SATOCM/ DM(3), RE(3), BETA(3)                                  
C                                                                               
         COMMON /PRECM/  Z(3),Q(3),XJ(3),DQ(3,3),DJ(3,3),                       
     +                   DZ(3,3),RR(3),A1,A2,A,B,C,D,F,G,                       
     +                   BET,BETP,ZFH0,ZHH0,ZHHST,ZA,ZB,                        
     +                   DEG1,DEG2,DEG3,DEG4,DEG3P,C1,C2,                       
     +                   C3,C4,C5,C6,C3C,ALF,ALFP,ALFT,                         
     +                   P3C,Q3C,ZHHDIF, OMP3C,OMQ3C,HA2,G2                     
C                                                                               
C   Initialize the flags and the I/O unit numbers for the potential             
C                                                                               
      DATA NASURF /1,35*0/                                                      
      DATA NDER /0/                                                             
         DATA NFLAG /1,1,15*0,6,0,0/                                            
C                                                                               
      DATA ANUZERO /0.0D0/                                                      
      DATA ICARTR,MSURF,MDER/3,0,1/                                             
      DATA NULBL /25*0/                                                         
      DATA NATOMS /3/                                                           
C                                                                               
C   Initialize the potential parameters; the energy parameters are in           
C   kcal/mol, and the lengths are in Angstroms.                                 
C                                                                               
         DATA DM /141.196D0,109.449D0,141.196D0/                                
         DATA RE /0.9170D0,0.7419D0,0.9170D0/                                   
         DATA BETA /2.2187D0, 1.9420D0, 2.2187D0/                               
C                                                                               
      DATA A1,A2,A,B,C,D,F,G                                                    
     +     /0.0395D0,0.201D0,1.26D0,1.60D0,0.0D0,0.0D0,                         
     +      0.0D0,0.2D0/                                                        
      DATA BET,BETP,ZFH0,ZHH0,ZHHST                                             
     +     /-0.101168D0,-0.46183D0,0.1325D0,                                    
     +       0.120D0,0.23107D0/                                                 
      DATA ZA,ZB                                                                
     +     /0.07464D0,-0.0445353D0/                                             
      DATA DEG1,DEG2,DEG3,DEG4,DEG3P                                            
     +     /10.0D0,20.0D0,60.0D0,90.0D0,41.0D0/                                 
      DATA C1,C2,C3,C4,C5,C6                                                    
     +     /1.15D-4,-6.0D-6,0.016D0,0.0005D0,0.00069D0,                         
     +      3.308D-2/                                                           
      DATA C3C,ALF,ALFP,ALFT,P3C,Q3C                                            
     +     /0.205081D0,0.18D0,0.078D0,2.14D0,                                   
     +      0.95D0,0.48D0/                                                      
C                                                                               
         END                                                                    
C                                                                               
C*****                                                                          


