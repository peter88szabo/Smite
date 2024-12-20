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
C   System:          ClHCl                                                      
C   Functional form: LEPS (London-Eyring-Polanyi-Sato)                          
C   Common name:     BCMR                                                       
C   Reference:       D. K. Bondi, J. N. L. Connor, J. Manz, and J. Romelt       
C                    J. Mol. Phys. 50, 467 (1983)                               
C                                                                               
C   Calling Sequence:                                                           
C      PREPOT - initializes the potential's variables and                       
C               must be called once before any calls to POT                     
C      POT    - driver for the evaluation of the energy and the derivatives     
C               of the energy with respect to the coordinates for a given       
C               geometric configuration                                         
C                                                                               
C   Units:                                                                      
C      energies    - hartrees                                                   
C      coordinates - bohrs                                                      
C      derivatives - hartrees/bohr                                              
C                                                                               
C   Surfaces:                                                                   
C      ground electronic state                                                  
C                                                                               
C   Zero of energy:                                                             
C      The classical potential energy is set equal to zero for the Cl           
C      infinitely far from the HCl diatomic and R(HCl) set equal to the         
C      HCl equilibrium diatomic value.                                          
C                                                                               
C   Parameters:                                                                 
C      Set in the BLOCK DATA subprogram PTPACM                                  
C                                                                               
C   Coordinates:                                                                
C      Internal, Definition: R(1) = R(first Cl-H)                               
C                            R(2) = R(H-second Cl)                              
C                            R(3) = R(first Cl-second Cl)                       
C                                                                               
C   Common Blocks (used between the calling program and this potential):        
C        passes the coordinates, energy, and derivatives of                     
C        the energy with respect to the coordinates.                            
C        passes the control flags where                                         
C        NDER  = 0 => no derivatives are calculated                             
C        NDER  = 1 => calculate first derivatives of the energy for the         
C                     ground electronic state with respect to the coordinates   
C        NFLAG  - Control flags                                                 
C      NFLAG(18-20)                                                             
C        passes the FORTRAN unit number used for POTENTIAL output               
C      /ASYCM/ EASYAB, EASYBC, EASYAC                                           
C        passes the energy in the three asymptotic valleys for an A+BC system.  
C        The energy in the AB valley, EASYAB, is equal to the energy            
C        of the C atom "infinitely" far from the AB diatomic and R(AB) set      
C        equal to Re(AB), the equilibrium bond length for the AB diatomic.      
C        In this potential the AB valley represents Cl infinitely far from      
C        the ClH diatomic and R(ClH) equal to Re(ClH).                          
C        Similarly, the terms EASYBC and EASYAC represent the energies in the   
C        other HCl and the ClCl asymptotic valleys, respectively.               
C                                                                               
C   Default Parameter Values:                                                   
C      Variable      Default value                                              
C      NDER             1                                                       
C      NFLAG(18)      6                                                         
C                                                                               
C*****                                                                          
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
         PARAMETER (R2     = 1.41421356D0)                                      
         PARAMETER (CKCAU  = 627.5095D0)                                        
         PARAMETER (CANGAU = 0.52917706D0)                                      
         COMMON /SATOCM/ D(3), RE(3), BETA(3), Z(3)                             
C                                                                               
         COMMON /POTCM/ ZPO(3), OP3Z(3), ZP3(3),TZP3(3),TOP3Z(3),               
     +                  DO4Z(3),B(3),X(3),COUL(3),EXCH(3)                       
                                                                                
C      DIMENSION ZPO(3), OP3Z(3), ZP3(3),TZP3(3), TOP3Z(3), DO4Z(3), B(3)       
C      DIMENSION X(3),COUL(3),EXCH(3)                                           
C                                                                               
      IF(NATOMS.GT.25) THEN                                                     
         WRITE(NFLAG(18),1111)                                                  
 1111    FORMAT(2X,'STOP. NUMBER OF ATOMS EXCEEDS ARRAY DIMENSIONS')            
         STOP                                                                   
      END IF                                                                    
C                                                                               
         WRITE (NFLAG(18), 600) D, RE, BETA, Z                                  
600   FORMAT(/,1X,'*****',1X,'Potential Energy Surface',1X,'*****',             
     *      //,1X,T5,'ClHCl BCMR LEPS potential energy surface',                
     *       //,1X,T5,'Parameters:',                                            
     *        /,1X,T5,'Bond', T46, 'Cl-H', T58, 'H-Cl', T68, 'Cl-Cl',           
     *        /,1X,T5,'Dissociation energies (kcal/mol):',                      
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,1X,T5,'Equilibrium bond lengths (Angstroms):',                  
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,1X,T5,'Morse beta parameters (Angstroms**-1):',                 
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,1X,T5,'Sato parameters:',                                       
     *        T44, F10.5, T55, F10.5, T66, F10.5,//,1X,'*****')                 
C                                                                               
C   CONVERT TO ATOMIC UNITS                                                     
C                                                                               
      DO  10 I = 1,3                                                            
      D(I)=D(I)/CKCAU                                                           
      RE(I) = RE(I)/CANGAU                                                      
      BETA(I) = BETA(I)*CANGAU                                                  
C                                                                               
C   COMPUTE USEFUL CONSTANTS                                                    
C                                                                               
      ZPO(I) = 1.0D0 + Z(I)                                                     
      OP3Z(I) = 1.0D0 + 3.0D0*Z(I)                                              
      TOP3Z(I) = 2.0D0*OP3Z(I)                                                  
      ZP3(I) = Z(I) + 3.0D0                                                     
      TZP3(I) = 2.0D0*ZP3(I)                                                    
      DO4Z(I) = D(I)/4.0D0/ZPO(I)                                               
      B(I) = BETA(I)*DO4Z(I)*2.0D0                                              
 10   CONTINUE                                                                  
C                                                                               
C   10 B(I) = BETA(I)*DO4Z(I)*2.0D0                                             
C                                                                               
C   Initialize the energy in the three asymptotic valleys                       
C                                                                               
      EASYAB = D(1)                                                             
      EASYBC = D(2)                                                             
      EASYAC = D(3)                                                             
C                                                                               
C                                                                               
      EZERO(1)=D(2)                                                             
C                                                                               
       DO I=1,5                                                                 
          REF(I) = ' '                                                          
       END DO                                                                   
C                                                                               
       REF(1)='D. K. Bondi, J. N. L. Connor, J. Manz, J. Romelt'                
       REF(2)='J. Mol. Phys. 50, 467(1983)'                                     
       REF(3)='S. Sato'                                                         
       REF(4)='J. Chem. Phys. 592, 2465(1955)'                                  
C                                                                               
      INDEXES(1) = 17                                                           
      INDEXES(2) = 1                                                            
      INDEXES(3) = 17                                                           
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
C   System:          ABC                                                        
C   Functional form: LEPS (London-Erying-Polyani-Sato)                          
C   Reference:       S. Sato                                                    
C                    J. Chem. Phys. 592, 2465 (1955)                            
C                                                                               
C   Coordinates:                                                                
C      Internal, Definition: R(1) = R(first Cl-H)                               
C                            R(2) = R(H-second Cl)                              
C                            R(3) = R(first Cl-second Cl)                       
C                                                                               
C        The energy in the AB valley, EASYAB, is equal to the energy            
C        of the C atom "infinitely" far from the AB diatomic and R(AB) set      
C        equal to Re(AB), the equilibrium bond length for the AB diatomic.      
C        In this potential the AB valley represents Cl infinitely far from      
C        the ClH diatomic and R(ClH) equal to Re(ClH).                          
C        Similarly, the terms EASYBC and EASYAC represent the energies in the   
C        other HCl and the ClCl asymptotic valleys, respectively.               
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
         PARAMETER (R2     = 1.41421356D0)                                      
         PARAMETER (CKCAU  = 627.5095D0)                                        
         PARAMETER (CANGAU = 0.52917706D0)                                      
         COMMON /SATOCM/ D(3), RE(3), BETA(3), Z(3)                             
C                                                                               
         COMMON /POTCM/ ZPO(3), OP3Z(3), ZP3(3),TZP3(3),TOP3Z(3),               
     +                  DO4Z(3),B(3),X(3),COUL(3),EXCH(3)                       
C                                                                               
      CALL CARTOU                                                               
      CALL CARTTOR                                                              
C                                                                               
C   Check the value of NDER                                                     
C                                                                               
         IF (NDER .GT. 1) THEN                                                  
             WRITE (NFLAG(18), 900) NDER                                        
             STOP 'POT 1'                                                       
         ENDIF                                                                  
C                                                                               
      DO 20 I = 1,3                                                             
            X(I)    = EXP(-BETA(I)*(R(I)-RE(I)))                                
            COUL(I) = DO4Z(I)*(ZP3(I)*X(I)-TOP3Z(I))*X(I)                       
            EXCH(I) = DO4Z(I)*(OP3Z(I)*X(I)-TZP3(I))*X(I)                       
20    CONTINUE                                                                  
      RAD = SQRT((EXCH(1)-EXCH(2))**2+(EXCH(2)-EXCH(3))**2+                     
     1      (EXCH(3)-EXCH(1))**2)                                               
      ENGYGS = COUL(1) + COUL(2) + COUL(3) - (RAD/R2) +                         
     +         EZERO(1) - ANUZERO                                               
C                                                                               
C   Compute the derivatives of the LEPS energy w.r.t.the coordinates            
C                                                                               
      IF (NDER .EQ. 1) THEN                                                     
          SVAL = EXCH(1) + EXCH(2) + EXCH(3)                                    
          DO 30 I = 1,3                                                         
                DEGSDR(I)=0.0D0                                                 
                IF(X(I).LT.1.0D-30) GO TO 30                                    
                TVAL = (3.0D0*EXCH(I)-SVAL)/R2*(OP3Z(I)*X(I)-ZP3(I))            
                IF(ABS(RAD) .LT. 1.0D-32 .AND.                                  
     1             ABS(TVAL) .GT. 1.0D-12) THEN                                 
                   WRITE(NFLAG(18), 6000) TVAL, RAD                             
                ELSE IF(ABS(RAD).GT.1.0D-32) THEN                               
                   TVAL = TVAL/RAD                                              
                END IF                                                          
                DEGSDR(I)=B(I)*X(I)*(TVAL-ZP3(I)*X(I)+OP3Z(I))                  
30        CONTINUE                                                              
      ENDIF                                                                     
C                                                                               
600   FORMAT(/,1X,'*****',1X,'Potential Energy Surface',1X,'*****',             
     *      //,1X,T5,'ClHCl BCMR LEPS potential energy surface',                
     *       //,1X,T5,'Parameters:',                                            
     *        /,1X,T5,'Bond', T46, 'Cl-H', T58, 'H-Cl', T68, 'Cl-Cl',           
     *        /,1X,T5,'Dissociation energies (kcal/mol):',                      
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,1X,T5,'Equilibrium bond lengths (Angstroms):',                  
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,1X,T5,'Morse beta parameters (Angstroms**-1):',                 
     *        T44, F10.5, T55, F10.5, T66, F10.5,                               
     *        /,1X,T5,'Sato parameters:',                                       
     *        T44, F10.5, T55, F10.5, T66, F10.5,//,1X,'*****')                 
900   FORMAT(/,1X,T5,'Error: POT has been called with NDER = ', I5,             
     *       /,1X,T12,'only the first derivatives, NDER = 1, are ',             
     *                'coded in this potential')                                
6000  FORMAT(/,1X,'In the ClHCl potential BCMR, T, RAD = ',1P,2E15.7,           
     1       /,1X,'T/RAD has been set equal to T')                              
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
         COMMON /SATOCM/ D(3), RE(3), BETA(3), Z(3)                             
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
         DATA D    / 106.477D0, 106.477D0, 57.983D0/                            
         DATA RE   / 1.275D0, 1.275D0, 1.988D0/                                 
         DATA BETA / 1.868D0, 1.868D0, 2.002D0/                                 
         DATA Z    / 0.115D0, 0.115D0, 0.115D0/                                 
C                                                                               
         END                                                                    
C                                                                               
C*****                                                                          
