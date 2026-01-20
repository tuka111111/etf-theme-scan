# Step6/7/10 Launchd

## 配置
```
cp launchd/com.you.stock.step6_7_10.daily.plist ~/Library/LaunchAgents/
```

## 起動
```
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.you.stock.step6_7_10.daily.plist
```

## 状態確認
```
launchctl print gui/$(id -u)/com.you.stock.step6_7_10.daily
```

## 停止
```
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.you.stock.step6_7_10.daily.plist
```

## ログ確認
```
tail -n 200 /Users/tupyon/auto_game_play_M4/stock/logs/launchd/step6_7_10_daily.err.log
tail -n 200 /Users/tupyon/auto_game_play_M4/stock/logs/launchd/step6_7_10_daily.out.log
```

## 手動疎通
```
bash /Users/tupyon/auto_game_play_M4/stock/script/job_step6_7_10_daily.sh
```
