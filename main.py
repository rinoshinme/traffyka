from vortex.traffic.video_analyzer import VideoAnalyzer


if __name__ == '__main__':
    video_file = 'D:/workspace/traffic/data/Relaxing highway traffic.mp4'
    cfg_file = 'cfg/demo.json'
    analyzer = VideoAnalyzer(cfg_file)
    ret = analyzer.open(video_file)
    if ret:
        analyzer.run()
