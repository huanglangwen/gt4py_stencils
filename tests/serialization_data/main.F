program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use gfdl_cloud_microphys_mod, only : gfdl_cloud_microphys_init, &
        gfdl_cloud_microphys_driver
#ifdef USE_MPIF90
    use mpi
    implicit none
#else
    implicit none
    include "mpif.h"
#endif
#define REPS 0
#define WRITE_DATA

    integer :: iie, kke, kbot, seconds

    real (kind=kind_phys) :: dt_in
    real (kind=kind_phys), dimension(:, :), allocatable :: area, land, &
        rain, snow, ice, graupel, rain_ref, snow_ref, ice_ref, graupel_ref
    real (kind=kind_phys), dimension(:, :, :), allocatable :: delp, dz, &
        uin, vin, p, pt, qv, ql, qr, qg, qa, qn, qi, qs, pt_dt, qa_dt, &
        udt, vdt, w, qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, refl_10cm, &
        qi_ref, qs_ref, qv_dt_ref, ql_dt_ref, qr_dt_ref, qi_dt_ref, &
        qs_dt_ref, qg_dt_ref, qa_dt_ref, pt_dt_ref, w_ref, udt_ref, vdt_ref, &
        refl_10cm_ref

    logical :: lradar, reset

    integer :: nunit, tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=100) :: nml_file
    character(len=6) :: ser_count_str

    ! GT4Py development
    logical :: WRITE_DATA_L=.false.
    logical :: VALIDATION=.false., BENCHMARK=.false., &
               NORMAL=.false., WEAK=.false., STRONG=.false.
    
    integer :: ierror, n_ranks, rank_idx
    integer :: factors_size, fact_idx, rep_idx
    
    real, dimension(:, :), allocatable :: timings
    
    real, dimension(:), allocatable :: factors
    
    real :: t_driver_start, t_driver_end, &
            t_tot_start, t_tot_end
    
    integer :: i_dim, iie_s, st, en, part
    real :: factor, divider, multiplier
    real (kind=kind_phys), dimension(:, :, :), allocatable :: qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
                                                              pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
                                                              qi_s, qs_s, &
                                                              qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
                                                              pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s
    real (kind=kind_phys), dimension(:, :), allocatable :: area_s, land_s, &
                                                           rain_s, snow_s, ice_s, graupel_s

    call MPI_INIT(ierror)
    
    call MPI_COMM_SIZE(MPI_COMM_WORLD, n_ranks, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank_idx, ierror)
    
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    if (rank_idx .eq. 0) call cpu_time(t_tot_start)

    nunit = 42
    nml_file = 'input.nml'
    ser_count_max = 10
    num_tiles = 6

#ifdef WRITE_DATA
    WRITE_DATA_L=.true.
#endif

#ifdef RUN_MODE
    if (RUN_MODE .eq. 0) then
        VALIDATION = .true.
    elseif (RUN_MODE .eq. 1) then
        BENCHMARK = .true.
        NORMAL    = .true.
    elseif (RUN_MODE .eq. 2) then
        BENCHMARK = .true.
        WEAK      = .true.
    elseif (RUN_MODE .eq. 3) then
        BENCHMARK = .true.
        STRONG    = .true.
    else
        VALIDATION = .true.
    endif
#endif

    if (NORMAL) then
        allocate(factors(4), timings(7, 12))
        factors = (/ 1./12., 1./6., 0.25, 1./3., 5./12., 0.5, 7./12., 2./3., 0.75, 10./12., 11./12., 1. /)
    elseif (WEAK) then
        allocate(factors(1), timings(7, 1))
        factors = (/ 1. /)
    elseif (STRONG) then
        allocate(factors(5), timings(7, 1))
        factors = (/ 12./n_ranks /)
    else
        allocate(factors(1), timings(7, 1))
        factors = (/ 1. /)
    endif
    
    timings = 0.
    
    factors_size = size(factors)
    
#ifdef PROGRESS_MODE
    call cdstart(num_tiles * ser_count_max)
#endif
    
    call gfdl_cloud_microphys_init(rank_idx, 0, nunit, (/''/), 6, nml_file)

    do tile = 0, num_tiles - 1
    
    !$ser verbatim if (WRITE_DATA_L) then
    !$ser init directory='./data_ref' prefix='Serialized' directory_ref='./data' prefix_ref='Generator' mpi_rank=tile
    !$ser verbatim else
    !$ser init directory='./data' prefix='Serialized' prefix_ref='Generator' mpi_rank=tile
    !$ser verbatim endif

    do ser_count = 0, ser_count_max

#ifdef PROGRESS_MODE
        call cdshow(tile * ser_count_max + ser_count)
#endif
        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser savepoint "cloud_mp-in-"//trim(ser_count_str)
        !$ser data iie=iie kke=kke kbot=kbot

        allocate(area(iie, 1), land(iie, 1), rain(iie, 1), snow(iie, 1), &
            ice(iie, 1), graupel(iie, 1), delp(iie, 1, kke), dz(iie, 1, kke), &
            uin(iie, 1, kke), vin(iie, 1, kke), p(iie, 1, kke), pt(iie, 1, kke), &
            qv(iie, 1, kke), ql(iie, 1, kke), qr(iie, 1, kke), qg(iie, 1, kke), &
            qa(iie, 1, kke), qn(iie, 1, kke), qi(iie, 1, kke), qs(iie, 1, kke), &
            pt_dt(iie, 1, kke), qa_dt(iie, 1, kke), udt(iie, 1, kke), vdt(iie, 1, kke), &
            w(iie, 1, kke), qv_dt(iie, 1, kke), ql_dt(iie, 1, kke), qr_dt(iie, 1, kke), &
            qi_dt(iie, 1, kke), qs_dt(iie, 1, kke), qg_dt(iie, 1, kke), refl_10cm(iie, 1, kke), &
            qi_ref(iie, 1, kke), qs_ref(iie, 1, kke), qv_dt_ref(iie, 1, kke), &
            ql_dt_ref(iie, 1, kke), qr_dt_ref(iie, 1, kke), qi_dt_ref(iie, 1, kke), &
            qs_dt_ref(iie, 1, kke), qg_dt_ref(iie, 1, kke), qa_dt_ref(iie, 1, kke), &
            pt_dt_ref(iie, 1, kke), w_ref(iie, 1, kke), udt_ref(iie, 1, kke), vdt_ref(iie, 1, kke), &
            rain_ref(iie, 1), snow_ref(iie, 1), ice_ref(iie, 1), graupel_ref(iie, 1), &
            refl_10cm_ref(iie, 1, kke))

        ! input
        !$ser data qv=qv ql=ql qr=qr qg=qg qa=qa qn=qn
        !$ser data pt=pt uin=uin vin=vin dz=dz delp=delp
        !$ser data area=area dt_in=dt_in land=land
        !$ser data seconds=seconds p=p lradar=lradar
        !$ser data reset=reset
        ! inout
        !$ser data qi=qi qs=qs qv_dt=qv_dt
        !$ser data ql_dt=ql_dt qr_dt=qr_dt qi_dt=qi_dt qs_dt=qs_dt qg_dt=qg_dt qa_dt=qa_dt
        !$ser data pt_dt=pt_dt w=w udt=udt vdt=vdt
        !$ser data rain=rain snow=snow ice=ice graupel=graupel
        !$ser data refl_10cm=refl_10cm
        
        do fact_idx=1, factors_size
            
            factor = factors(fact_idx)
            iie_s  = iie * factor
            
            allocate( qv_s(iie_s, 1, kke), ql_s(iie_s, 1, kke), qr_s(iie_s, 1, kke), qg_s(iie_s, 1, kke), qa_s(iie_s, 1, kke), qn_s(iie_s, 1, kke), &
                      pt_s(iie_s, 1, kke), uin_s(iie_s, 1, kke), vin_s(iie_s, 1, kke), dz_s(iie_s, 1, kke), delp_s(iie_s, 1, kke), p_s(iie_s, 1, kke), &
                      area_s(iie_s, 1), land_s(iie_s, 1), &
                      qi_s(iie_s, 1, kke), qs_s(iie_s, 1, kke), &
                      qv_dt_s(iie_s, 1, kke), ql_dt_s(iie_s, 1, kke), qr_dt_s(iie_s, 1, kke), qi_dt_s(iie_s, 1, kke), qs_dt_s(iie_s, 1, kke), qg_dt_s(iie_s, 1, kke), qa_dt_s(iie_s, 1, kke), &
                      pt_dt_s(iie_s, 1, kke), w_s(iie_s, 1, kke), udt_s(iie_s, 1, kke), vdt_s(iie_s, 1, kke), refl_10cm_s(iie_s, 1, kke), &
                      rain_s(iie_s, 1), snow_s(iie_s, 1), ice_s(iie_s, 1), graupel_s(iie_s, 1) )
            
            if (factor .le. 1.) then
                
                qv_s(1:iie_s, :, :)   = qv(1:iie_s, :, :)
                ql_s(1:iie_s, :, :)   = ql(1:iie_s, :, :)
                qr_s(1:iie_s, :, :)   = qr(1:iie_s, :, :)
                qg_s(1:iie_s, :, :)   = qg(1:iie_s, :, :)
                qa_s(1:iie_s, :, :)   = qa(1:iie_s, :, :)
                qn_s(1:iie_s, :, :)   = qn(1:iie_s, :, :)
                pt_s(1:iie_s, :, :)   = pt(1:iie_s, :, :)
                uin_s(1:iie_s, :, :)  = uin(1:iie_s, :, :)
                vin_s(1:iie_s, :, :)  = vin(1:iie_s, :, :)
                dz_s(1:iie_s, :, :)   = dz(1:iie_s, :, :)
                delp_s(1:iie_s, :, :) = delp(1:iie_s, :, :)
                p_s(1:iie_s, :, :)    = p(1:iie_s, :, :)
                area_s(1:iie_s, :)    = area(1:iie_s, :)
                land_s(1:iie_s, :)    = land(1:iie_s, :)
            
            else
                
                do part=1, int(factor)
                    
                    st = (part-1)*iie + 1
                    en = part*iie
                    
                    qv_s(st:en, :, :)   = qv(1:iie, :, :)
                    ql_s(st:en, :, :)   = ql(1:iie, :, :)
                    qr_s(st:en, :, :)   = qr(1:iie, :, :)
                    qg_s(st:en, :, :)   = qg(1:iie, :, :)
                    qa_s(st:en, :, :)   = qa(1:iie, :, :)
                    qn_s(st:en, :, :)   = qn(1:iie, :, :)
                    pt_s(st:en, :, :)   = pt(1:iie, :, :)
                    uin_s(st:en, :, :)  = uin(1:iie, :, :)
                    vin_s(st:en, :, :)  = vin(1:iie, :, :)
                    dz_s(st:en, :, :)   = dz(1:iie, :, :)
                    delp_s(st:en, :, :) = delp(1:iie, :, :)
                    p_s(st:en, :, :)    = p(1:iie, :, :)
                    area_s(st:en, :)    = area(1:iie, :)
                    land_s(st:en, :)    = land(1:iie, :)
                    
                enddo
                
            endif
            
            do rep_idx=1, REPS+1
                
                if (factor .le. 1.) then
                    
                    qi_s(1:iie_s, :, :)        = qi(1:iie_s, :, :)
                    qs_s(1:iie_s, :, :)        = qs(1:iie_s, :, :)
                    qv_dt_s(1:iie_s, :, :)     = qv_dt(1:iie_s, :, :)
                    ql_dt_s(1:iie_s, :, :)     = ql_dt(1:iie_s, :, :)
                    qr_dt_s(1:iie_s, :, :)     = qr_dt(1:iie_s, :, :)
                    qi_dt_s(1:iie_s, :, :)     = qi_dt(1:iie_s, :, :)
                    qs_dt_s(1:iie_s, :, :)     = qs_dt(1:iie_s, :, :)
                    qg_dt_s(1:iie_s, :, :)     = qg_dt(1:iie_s, :, :)
                    qa_dt_s(1:iie_s, :, :)     = qa_dt(1:iie_s, :, :)
                    pt_dt_s(1:iie_s, :, :)     = pt_dt(1:iie_s, :, :)
                    w_s(1:iie_s, :, :)         = w(1:iie_s, :, :)
                    udt_s(1:iie_s, :, :)       = udt(1:iie_s, :, :)
                    vdt_s(1:iie_s, :, :)       = vdt(1:iie_s, :, :)
                    refl_10cm_s(1:iie_s, :, :) = refl_10cm(1:iie_s, :, :)
                    rain_s(1:iie_s, :)         = rain(1:iie_s, :)
                    snow_s(1:iie_s, :)         = snow(1:iie_s, :)
                    ice_s(1:iie_s, :)          = ice(1:iie_s, :)
                    graupel_s(1:iie_s, :)      = graupel(1:iie_s, :)
                
                else
                    
                    do part=1, int(factor)
                        
                        st = (part-1)*iie + 1
                        en = part*iie
                         
                        qi_s(st:en, :, :)        = qi(1:iie, :, :)
                        qs_s(st:en, :, :)        = qs(1:iie, :, :)
                        qv_dt_s(st:en, :, :)     = qv_dt(1:iie, :, :)
                        ql_dt_s(st:en, :, :)     = ql_dt(1:iie, :, :)
                        qr_dt_s(st:en, :, :)     = qr_dt(1:iie, :, :)
                        qi_dt_s(st:en, :, :)     = qi_dt(1:iie, :, :)
                        qs_dt_s(st:en, :, :)     = qs_dt(1:iie, :, :)
                        qg_dt_s(st:en, :, :)     = qg_dt(1:iie, :, :)
                        qa_dt_s(st:en, :, :)     = qa_dt(1:iie, :, :)
                        pt_dt_s(st:en, :, :)     = pt_dt(1:iie, :, :)
                        w_s(st:en, :, :)         = w(1:iie, :, :)
                        udt_s(st:en, :, :)       = udt(1:iie, :, :)
                        vdt_s(st:en, :, :)       = vdt(1:iie, :, :)
                        refl_10cm_s(st:en, :, :) = refl_10cm(1:iie, :, :)
                        rain_s(st:en, :)         = rain(1:iie, :)
                        snow_s(st:en, :)         = snow(1:iie, :)
                        ice_s(st:en, :)          = ice(1:iie, :)
                        graupel_s(st:en, :)      = graupel(1:iie, :)
                        
                    enddo
                    
                endif
            
                if (BENCHMARK .and. (rep_idx .gt. 1)) call cpu_time(t_driver_start)

                call gfdl_cloud_microphys_driver (qv_s, ql_s, qr_s, qi_s, qs_s, qg_s, qa_s, qn_s,     &
                    qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, pt_dt_s, pt_s, w_s,    &
                    uin_s, vin_s, udt_s, vdt_s, dz_s, delp_s, area_s, dt_in, land_s, rain_s, snow_s, ice_s, &
                    graupel_s, .false., .true., 1, iie_s, 1, 1, 1,                        &
                    kke, 1, kbot, seconds, p_s, lradar, refl_10cm_s, reset, (BENCHMARK .and. (rep_idx .gt. 1)), timings, fact_idx)

                if (BENCHMARK .and. (rep_idx .gt. 1)) then
                    
                    call cpu_time(t_driver_end)
                
                    timings(6, fact_idx) = timings(6, fact_idx) + (t_driver_end - t_driver_start)
                
                endif
                
            enddo
            
            if (fact_idx .ne. factors_size) then
                
                deallocate( qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
                            pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
                            area_s, land_s, &
                            qi_s, qs_s, &
                            qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
                            pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s, &
                            rain_s, snow_s, ice_s, graupel_s )
            
            endif
            
        enddo
        
        qi        = qi_s
        qs        = qs_s
        qv_dt     = qv_dt_s
        ql_dt     = ql_dt_s
        qr_dt     = qr_dt_s
        qi_dt     = qi_dt_s
        qs_dt     = qs_dt_s
        qg_dt     = qg_dt_s
        qa_dt     = qa_dt_s
        pt_dt     = pt_dt_s
        w         = w_s
        udt       = udt_s
        vdt       = vdt_s
        refl_10cm = refl_10cm_s
        rain      = rain_s
        snow      = snow_s
        ice       = ice_s
        graupel   = graupel_s
        
        !$ser mode read
        !$ser savepoint "cloud_mp-out-"//trim(ser_count_str)
        ! inout
        !$ser data qi=qi_ref qs=qs_ref qv_dt=qv_dt_ref
        !$ser data ql_dt=ql_dt_ref qr_dt=qr_dt_ref qi_dt=qi_dt_ref qs_dt=qs_dt_ref qg_dt=qg_dt_ref qa_dt=qa_dt_ref
        !$ser data pt_dt=pt_dt_ref w=w_ref udt=udt_ref vdt=vdt_ref
        !$ser data rain=rain_ref snow=snow_ref ice=ice_ref graupel=graupel_ref
        !$ser data refl_10cm=refl_10cm_ref
        
        !$ser verbatim if (WRITE_DATA_L) then
        !$ser mode write
        !$ser savepoint "cloud_mp-out-x-"//trim(ser_count_str)
        !$ser data qi=qi qs=qs qv_dt=qv_dt
        !$ser data ql_dt=ql_dt qr_dt=qr_dt qi_dt=qi_dt qs_dt=qs_dt qg_dt=qg_dt qa_dt=qa_dt
        !$ser data pt_dt=pt_dt w=w udt=udt vdt=vdt
        !$ser data rain=rain snow=snow ice=ice graupel=graupel
        !$ser data refl_10cm=refl_10cm
        !$ser verbatim endif

        if (VALIDATION) then
        
            call check_r_3d(qi, qi_ref, "qi")
            call check_r_3d(qs, qs_ref, "qs")
            call check_r_3d(qv_dt, qv_dt_ref, "qv_dt")
            call check_r_3d(ql_dt, ql_dt_ref, "ql_dt")
            call check_r_3d(qr_dt, qr_dt_ref, "qr_dt")
            call check_r_3d(qi_dt, qi_dt_ref, "qi_dt")
            call check_r_3d(qs_dt, qs_dt_ref, "qs_dt")
            call check_r_3d(qg_dt, qg_dt_ref, "qg_dt")
            call check_r_3d(qa_dt, qa_dt_ref, "qa_dt")
            call check_r_3d(pt_dt, pt_dt_ref, "pt_dt")
            call check_r_3d(w, w_ref, "w")
            call check_r_3d(udt, udt_ref, "udt")
            call check_r_3d(vdt, vdt_ref, "vdt")
            call check_r_2d(rain, rain_ref, "rain")
            call check_r_2d(snow, snow_ref, "snow")
            call check_r_2d(ice, ice_ref, "ice")
            call check_r_2d(graupel, graupel_ref, "graupel")
            call check_r_3d(refl_10cm, refl_10cm_ref, "refl_10cm")
        
        endif
        
        deallocate(area, land, rain, snow, ice, graupel, delp, dz, uin, vin, &
            p, pt, qv, ql, qr, qg, qa, qn, qi, qs, pt_dt, qa_dt, udt, vdt, w, &
            qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, refl_10cm, qi_ref, &
            qs_ref, qv_dt_ref, ql_dt_ref, qr_dt_ref, qi_dt_ref, &
            qs_dt_ref, qg_dt_ref, qa_dt_ref, pt_dt_ref, w_ref, udt_ref, vdt_ref, &
            rain_ref, snow_ref, ice_ref, graupel_ref, &
            refl_10cm_ref, &
            qv_s, ql_s, qr_s, qg_s, qa_s, qn_s, &
            pt_s, uin_s, vin_s, dz_s, delp_s, p_s, &
            area_s, land_s, &
            qi_s, qs_s, &
            qv_dt_s, ql_dt_s, qr_dt_s, qi_dt_s, qs_dt_s, qg_dt_s, qa_dt_s, &
            pt_dt_s, w_s, udt_s, vdt_s, refl_10cm_s, &
            rain_s, snow_s, ice_s, graupel_s )

    end do
    
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    !$ser cleanup

    end do

#ifdef PROGRESS_MODE
    call cdstop()
#endif
    
    if (BENCHMARK) call MPI_REDUCE(timings(1:4, :), timings(1:4, :), 4*factors_size, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
    
    if (rank_idx .eq. 0) then
        
        call cpu_time(t_tot_end)
        
        timings(7, 1) = t_tot_end - t_tot_start
        
        write(*,*) 'FINISHED!'
        
        if (BENCHMARK) then
            
            if (NORMAL) then
                open(1, file="../out/timings_benchmark_fortran_normal.dat")
            else if (WEAK) then
                open(1, file="../out/timings_weak_fortran_normal.dat", status="old", position="append")
            else if (STRONG) then
                open(1, file="../out/timings_strong_fortran_normal.dat", status="old", position="append")
            endif
            
            timings(1:6, :) = timings(1:6, :) / REPS
            
            do fact_idx=1, factors_size
                
                i_dim = int(iie * factors(fact_idx) * n_ranks)
                
                timings(5, fact_idx) = timings(1, fact_idx) + timings(2, fact_idx) + timings(3, fact_idx) + timings(4, fact_idx)
                
                write(*,*) ""
                write(*,'(a, I5, a, I2, a)') "For gridsize ", i_dim, "x1x", kke, ":"
                write(*,'(a, F7.3, a)') "> Warm rain processes (1st pass) => ", timings(1, fact_idx), " seconds"
                write(*,'(a, F7.3, a)') "> Sedimentation                  => ", timings(2, fact_idx), " seconds"
                write(*,'(a, F7.3, a)') "> Warm rain processes (2nd pass) => ", timings(3, fact_idx), " seconds"
                write(*,'(a, F7.3, a)') "> Ice-phase microphysics         => ", timings(4, fact_idx), " seconds"
                write(*,'(a, F7.3, a)') "> Main loop                      => ", timings(5, fact_idx), " seconds"
                write(*,'(a, F7.3, a)') "> Whole driver                   => ", timings(6, fact_idx), " seconds"
                
                if (factors(fact_idx) .le. 1.) then
                    divider    = factors(fact_idx)
                    multiplier = 1.
                else
                    divider    = 1.
                    multiplier = factors(fact_idx)
                endif
                
                write(1, *) divider, multiplier, i_dim * kke, n_ranks, &
                            0., timings(1, fact_idx), 0., timings(2, fact_idx), &
                            0., timings(3, fact_idx), 0., timings(4, fact_idx), &
                            0., timings(5, fact_idx), timings(6, fact_idx)
                
            enddo
            
            close(1)
            
        endif
    
        write(*,*) ""
        write(*,'(a, F7.3, a)') ">> Total elapsed time: ", timings(7, 1), " seconds"
        
    endif
    
    call MPI_FINALIZE(ierror)

contains

    subroutine check_r_3d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:,:,:), b(:,:,:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_r_3d

    subroutine check_r_2d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_r_2d

    subroutine check_r_1d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:), b(:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_r_1d

    elemental logical function isclose_r(a, b, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a, b
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        real (kind=kind_phys) :: atol_local, rtol_local

        if (present(atol)) then
            atol_local = atol
        else
            atol_local = 1.0d-13
        end if
        if (present(rtol)) then
            rtol_local = rtol
        else
            rtol_local = 1.0d-11
        end if

        isclose_r = abs(a - b) <= (atol_local + rtol_local * abs(b))

    end function isclose_r

    elemental logical function isclose_i(a, b, atol)
        implicit none
        integer, intent(in) :: a, b
        integer, intent(in), optional :: atol

        integer :: atol_local, rtol_local

        if (present(atol)) then
            atol_local = atol
        else
            atol_local = 0
        end if

        isclose_i = abs(a - b) <= atol_local

    end function isclose_i

    subroutine tic(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t1)
    end subroutine tic

    subroutine toc(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t2)
        write(*,'(a,f5.3,a)') "    Time Taken --> ", 1000*real(t2-t1), ' ms'
    end subroutine toc

end program test
