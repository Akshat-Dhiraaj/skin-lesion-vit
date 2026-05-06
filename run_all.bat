@echo off
call conda activate skin-lesion
cd D:\ML\skin-lesion-vit

for /f "delims=" %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set TS=%%a
set LOGFILE=logs\run_%TS%.log

echo ======================================== > %LOGFILE%
echo  skin-lesion-vit Full Training Run >> %LOGFILE%
echo  Started: %date% %time% >> %LOGFILE%
echo ======================================== >> %LOGFILE%

echo Starting training run... Log: %LOGFILE%
echo.

echo. >> %LOGFILE%
echo === PHASE 1: ResNet50 Frozen === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 1: ResNet50 Frozen...
python src/engine.py --phase 1 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 1 done.

echo. >> %LOGFILE%
echo === PHASE 2: EfficientNetV2 === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 2: EfficientNetV2...
python src/engine.py --phase 2 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 2 done.

echo. >> %LOGFILE%
echo === PHASE 3: ViT-Base === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 3: ViT-Base...
python src/engine.py --phase 3 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 3 done.

echo. >> %LOGFILE%
echo === PHASE 4: DINOv2 === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 4: DINOv2...
python src/engine.py --phase 4 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 4 done.

echo. >> %LOGFILE%
echo === PHASE 5: DINOv2 Registers === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 5: DINOv2 Registers...
python src/engine.py --phase 5 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 5 done.

echo. >> %LOGFILE%
echo === PHASE 6: AIMv2 === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 6: AIMv2...
python src/engine.py --phase 6 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 6 done.

echo. >> %LOGFILE%
echo === PHASE 7: SigLIP === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 7: SigLIP...
python src/engine.py --phase 7 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 7 done.

echo. >> %LOGFILE%
echo === PHASE 10: DINOv2+Registers Full Fine-tune === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 10: DINOv2+Registers Full Fine-tune...
python src/engine.py --phase 10 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 10 done.

echo. >> %LOGFILE%
echo === PHASE 11: Re-Ensemble === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 11: Re-Ensemble...
python src/advanced_phases.py --phase 11 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 11 done.

echo. >> %LOGFILE%
echo === PHASE 12: Multimodal === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 12: Multimodal...
python src/advanced_phases.py --phase 12 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 12 done.

echo. >> %LOGFILE%
echo === PHASE 13: Focal Loss === >> %LOGFILE%
echo Started: %time% >> %LOGFILE%
echo Running Phase 13: Focal Loss...
python src/advanced_phases.py --phase 13 >> %LOGFILE% 2>&1
echo Finished: %time% >> %LOGFILE%
echo Phase 13 done.

echo. >> %LOGFILE%
echo ======================================== >> %LOGFILE%
echo  ALL PHASES COMPLETE >> %LOGFILE%
echo  Ended: %date% %time% >> %LOGFILE%
echo ======================================== >> %LOGFILE%

echo.
echo ========================================
echo  ALL PHASES COMPLETE
echo  Log saved to: %LOGFILE%
echo ========================================
pause