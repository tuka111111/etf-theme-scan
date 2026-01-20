# Step10 LaunchAgent

## 목적
매일 지정된 시간에 `run_pipeline.sh` → `step10_daily_runner.py` 를 연속 실행합니다.

## 설치
```
ln -s $HOME/auto_game_play_M4/stock $HOME/stock
cp launchd/com.you.stock.daily.plist ~/Library/LaunchAgents/
```

## 로드/언로드
```
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.you.stock.daily.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.you.stock.daily.plist
```

## 상태 확인
```
launchctl print gui/$(id -u)/com.you.stock.daily
```

## 로그 확인
```
tail -n 200 $HOME/stock/logs/launchd/daily.err.log
tail -n 200 $HOME/stock/logs/launchd/daily.out.log
```

## 수동 실행
```
bash $HOME/stock/script/job_daily.sh
```
