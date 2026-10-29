import { flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MSEPlayer, PlayerFactory, WebRTCPlayer } from '../PlayerFactory'

class FakeSocket {
  static instances: FakeSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()
  send = vi.fn()
  constructor(_url: string) { FakeSocket.instances.push(this) }
}

class FakePeer {
  static instances: FakePeer[] = []
  connectionState: RTCPeerConnectionState = 'new'
  ontrack: ((event: { streams: MediaStream[] }) => void) | null = null
  onconnectionstatechange: (() => void) | null = null
  onicecandidate: ((event: RTCPeerConnectionIceEvent) => void) | null = null
  close = vi.fn()
  addTransceiver = vi.fn()
  createOffer = vi.fn().mockResolvedValue({ type: 'offer', sdp: 'test' })
  setLocalDescription = vi.fn().mockResolvedValue(undefined)
  setRemoteDescription = vi.fn().mockResolvedValue(undefined)
  addIceCandidate = vi.fn().mockResolvedValue(undefined)
  constructor(_options: unknown) { FakePeer.instances.push(this) }
}

class FakeMediaSource extends EventTarget {
  static instances: FakeMediaSource[] = []
  static isTypeSupported = vi.fn<(type: string) => boolean>(() => true)
  readyState = 'closed'
  constructor() { super(); FakeMediaSource.instances.push(this) }
}

function stream() {
  const stop = vi.fn()
  return { media: { getTracks: () => [{ stop }] } as unknown as MediaStream, stop }
}

function decodedFrame(video: HTMLVideoElement) {
  for (const [key, value] of Object.entries({ readyState: 2, videoWidth: 640, videoHeight: 480, paused: false })) {
    Object.defineProperty(video, key, { value, configurable: true })
  }
  video.dispatchEvent(new Event('loadeddata'))
}

describe('player media ownership and cancellation', () => {
  const NativeURL = globalThis.URL
  let revoke: ReturnType<typeof vi.fn<(url: string) => void>>

  beforeEach(() => {
    FakeSocket.instances = []
    FakePeer.instances = []
    FakeMediaSource.instances = []
    FakeMediaSource.isTypeSupported.mockReset().mockReturnValue(true)
    revoke = vi.fn<(url: string) => void>()
    class TestURL extends NativeURL {
      static createObjectURL = vi.fn().mockReturnValue('blob:old-player')
      static revokeObjectURL = revoke
    }
    vi.stubGlobal('URL', TestURL)
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('RTCPeerConnection', FakePeer)
    vi.stubGlobal('MediaSource', FakeMediaSource)
    vi.stubGlobal('RTCSessionDescription', class { constructor(_init: unknown) {} })
    vi.stubGlobal('RTCIceCandidate', class { constructor(_init: unknown) {} })
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('stops only its own WebRTC tracks without detaching a newer owner', async () => {
    const video = document.createElement('video')
    const old = stream()
    const newer = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [old.media] })
    decodedFrame(video)
    await expect(pending).resolves.toBe(true)
    video.srcObject = newer.media

    player.destroy()

    expect(old.stop).toHaveBeenCalledOnce()
    expect(newer.stop).not.toHaveBeenCalled()
    expect(video.srcObject).toBe(newer.media)
  })

  it('requests recovery instead of reporting playing after a connected WebRTC socket error clears media', async () => {
    const video = document.createElement('video')
    const owned = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onStatusChange = vi.fn()
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [owned.media] })
    decodedFrame(video)
    await expect(pending).resolves.toBe(true)
    expect(player.status).toBe('playing')

    FakeSocket.instances[0]?.onerror?.()

    expect(player.status).toBe('error')
    expect(player.onStatusChange).toHaveBeenLastCalledWith('error')
    expect(player.onReconnectRequest).toHaveBeenCalledOnce()
    expect(video.srcObject).toBeNull()
    expect(owned.stop).toHaveBeenCalledOnce()
    player.destroy()
    player.destroy()
    expect(player.onReconnectRequest).toHaveBeenCalledOnce()
    expect(owned.stop).toHaveBeenCalledOnce()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles an initial WebRTC failure without requesting an established-player reconnect', async () => {
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onerror?.()
    await expect(pending).resolves.toBe(false)
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    player.destroy()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles a synchronous peer constructor failure immediately and clears its timeout', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('RTCPeerConnection', class {
      constructor() { throw new Error('WebRTC capability unavailable') }
    })
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    expect(vi.getTimerCount()).toBe(1)
    expect(() => FakeSocket.instances[0]?.onopen?.()).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    player.destroy()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles a synchronous transceiver failure and cleans the partially created peer', async () => {
    vi.useFakeTimers()
    class UnsupportedPeer extends FakePeer {
      addTransceiver = vi.fn(() => { throw new Error('unsupported transceiver') })
    }
    vi.stubGlobal('RTCPeerConnection', UnsupportedPeer)
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    expect(() => FakeSocket.instances[0]?.onopen?.()).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
    expect(FakePeer.instances[0]?.close).toHaveBeenCalledOnce()
    expect(FakePeer.instances[0]?.createOffer).not.toHaveBeenCalled()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
  })

  it.each(['answer', 'candidate'])('ignores a late rejected WebRTC %s after cancellation without disturbing the next media owner', async kind => {
    let rejectOperation!: (reason: Error) => void
    const operation = new Promise<void>((_resolve, reject) => { rejectOperation = reject })
    class SlowPeer extends FakePeer {
      setRemoteDescription = vi.fn().mockReturnValue(operation)
      addIceCandidate = vi.fn().mockReturnValue(operation)
    }
    vi.stubGlobal('RTCPeerConnection', SlowPeer)
    const video = document.createElement('video')
    const old = stream()
    const newer = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [old.media] })
    decodedFrame(video)
    await expect(pending).resolves.toBe(true)
    FakeSocket.instances[0]?.onmessage?.({ data: JSON.stringify({ type: `webrtc/${kind}`, value: 'negotiation' }) } as MessageEvent)
    player.destroy()
    video.srcObject = newer.media
    rejectOperation(new Error('peer was closed during negotiation'))
    await flushPromises()

    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    expect(video.srcObject).toBe(newer.media)
    expect(newer.stop).not.toHaveBeenCalled()
    expect(old.stop).toHaveBeenCalledOnce()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles a rejected initial remote description without an unhandled negotiation failure', async () => {
    class RejectedPeer extends FakePeer {
      setRemoteDescription = vi.fn().mockRejectedValue(new Error('invalid answer'))
    }
    vi.stubGlobal('RTCPeerConnection', RejectedPeer)
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakeSocket.instances[0]?.onmessage?.({ data: JSON.stringify({ type: 'webrtc/answer', value: 'bad-answer' }) } as MessageEvent)
    await expect(pending).resolves.toBe(false)
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    expect(FakePeer.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('requests recovery for a current ICE failure after playback has started', async () => {
    class RejectedPeer extends FakePeer {
      addIceCandidate = vi.fn().mockRejectedValue(new Error('candidate rejected'))
    }
    vi.stubGlobal('RTCPeerConnection', RejectedPeer)
    const video = document.createElement('video')
    const owned = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [owned.media] })
    decodedFrame(video)
    await expect(pending).resolves.toBe(true)
    FakeSocket.instances[0]?.onmessage?.({ data: JSON.stringify({ type: 'webrtc/candidate', value: 'bad-candidate' }) } as MessageEvent)
    await flushPromises()
    expect(player.status).toBe('error')
    expect(player.onReconnectRequest).toHaveBeenCalledOnce()
    expect(owned.stop).toHaveBeenCalledOnce()
  })

  it('fails malformed WebRTC messages through controlled cleanup', async () => {
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    expect(() => FakeSocket.instances[0]?.onmessage?.({ data: '{broken' } as MessageEvent)).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it.each(['failed', 'disconnected'] as const)('lets initial WebRTC %s fall through without restarting WebRTC', async state => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]!.onopen!()
    const peer = FakePeer.instances[0]!, owned = stream()
    peer.ontrack!({ streams: [owned.media] })
    expect(player.status).toBe('connecting')
    expect(vi.getTimerCount()).toBe(1)
    peer.connectionState = state
    peer.onconnectionstatechange!()
    await expect(pending).resolves.toBe(false)
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    expect(owned.stop).toHaveBeenCalledOnce()
    decodedFrame(video)
    expect(player.status).not.toBe('playing')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not count a signaled track without decoded frames as playback', async () => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    const pending = player.connect()
    FakeSocket.instances[0]!.onopen!()
    FakePeer.instances[0]!.ontrack!({ streams: [stream().media] })
    expect(player.status).toBe('connecting')
    await vi.advanceTimersByTimeAsync(15000)
    await expect(pending).resolves.toBe(false)
    expect(player.status).not.toBe('playing')
  })

  it('keeps paused metadata-only WebRTC connecting until real playback starts', async () => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    const pending = player.connect()
    FakeSocket.instances[0]!.onopen!()
    FakePeer.instances[0]!.ontrack!({ streams: [stream().media] })
    for (const [key, value] of Object.entries({ readyState: 2, videoWidth: 640, videoHeight: 480, paused: true })) {
      Object.defineProperty(video, key, { value, configurable: true })
    }
    video.dispatchEvent(new Event('loadeddata'))
    expect(player.status).toBe('connecting')
    decodedFrame(video)
    await expect(pending).resolves.toBe(true)
    expect(player.status).toBe('playing')
    expect(vi.getTimerCount()).toBe(0)
    player.destroy()
  })

  it('cleans up rejected WebRTC playback before accepting the first frame', async () => {
    const video = document.createElement('video')
    vi.mocked(video.play).mockRejectedValueOnce(new Error('playback denied'))
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]!.onopen!()
    const owned = stream()
    FakePeer.instances[0]!.ontrack!({ streams: [owned.media] })
    await expect(pending).resolves.toBe(false)
    expect(owned.stop).toHaveBeenCalledOnce()
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
  })

  it('releases only its MSE URL without resetting the next player element', async () => {
    const video = document.createElement('video')
    const load = vi.spyOn(video, 'load').mockImplementation(() => {})
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    expect(video.getAttribute('src')).toBe('blob:old-player')
    video.src = 'blob:new-player'
    player.destroy()

    await expect(pending).resolves.toBe(false)
    expect(video.getAttribute('src')).toBe('blob:new-player')
    expect(load).not.toHaveBeenCalled()
    expect(revoke).toHaveBeenCalledOnce()
    expect(revoke).toHaveBeenCalledWith('blob:old-player')
  })

  it('negotiates MSE codecs on socket open and does not subscribe a cancelled player', async () => {
    FakeMediaSource.isTypeSupported.mockImplementation(type => type.includes('avc1.64002A'))
    const video = document.createElement('video')
    vi.spyOn(video, 'load').mockImplementation(() => {})
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    expect(video.play).toHaveBeenCalledOnce()
    FakeMediaSource.instances[0]!.dispatchEvent(new Event('sourceopen'))
    const socket = FakeSocket.instances[0]!
    socket.onopen!()
    expect(socket.send).toHaveBeenCalledWith(JSON.stringify({ type: 'mse', value: 'avc1.64002A' }))
    const lateOpen = socket.onopen!
    player.destroy()
    await expect(pending).resolves.toBe(false)
    lateOpen()
    expect(socket.send).toHaveBeenCalledTimes(1)
  })

  it('cleans up rejected MSE playback instead of leaving a paused connecting stream', async () => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    vi.spyOn(video, 'load').mockImplementation(() => {})
    vi.mocked(video.play).mockRejectedValueOnce(new Error('playback denied'))
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    await expect(player.connect()).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
    expect(revoke).toHaveBeenCalledOnce()
    player.destroy()
  })

  it('ignores an old play rejection after another player owns the element', async () => {
    let reject!: (error: Error) => void
    const video = document.createElement('video')
    vi.spyOn(video, 'load').mockImplementation(() => {})
    vi.mocked(video.play).mockReturnValueOnce(new Promise((_, fail) => { reject = fail }))
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    player.destroy()
    await expect(pending).resolves.toBe(false)
    video.src = 'blob:new-owner'
    reject(new Error('old playback aborted'))
    await flushPromises()
    expect(video.getAttribute('src')).toBe('blob:new-owner')
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
  })

  it.each(['unsupported_codec', 'send_failure'])('fails MSE negotiation promptly on %s', async failure => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    vi.spyOn(video, 'load').mockImplementation(() => {})
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    FakeMediaSource.instances[0]!.dispatchEvent(new Event('sourceopen'))
    const socket = FakeSocket.instances[0]!
    if (failure === 'unsupported_codec') FakeMediaSource.isTypeSupported.mockReturnValue(false)
    else socket.send.mockImplementation(() => { throw new Error('socket send failed') })
    expect(() => socket.onopen!()).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(socket.close).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
    player.destroy()
  })

  it('does not clear a newer WebRTC stream even if its old MSE src attribute remains', async () => {
    const video = document.createElement('video')
    const load = vi.spyOn(video, 'load').mockImplementation(() => {})
    const newer = stream()
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    video.srcObject = newer.media
    player.destroy()

    await expect(pending).resolves.toBe(false)
    expect(video.srcObject).toBe(newer.media)
    expect(newer.stop).not.toHaveBeenCalled()
    expect(load).not.toHaveBeenCalled()
    expect(revoke).toHaveBeenCalledWith('blob:old-player')
  })

  it.each(['webrtc', 'mse'])('settles and clears the pending %s timeout on destroy', async (protocol) => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    vi.spyOn(video, 'load').mockImplementation(() => {})
    const player = protocol === 'webrtc'
      ? new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
      : new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    expect(vi.getTimerCount()).toBe(1)
    player.destroy()
    await expect(pending).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('aborts a pending factory connection instead of waiting for its network timeout', async () => {
    vi.spyOn(PlayerFactory, 'fetchStreamInfo').mockResolvedValue({
      camera_id: 'cam1', go2rtc: true, webrtc_ws: 'ws://example.test/webrtc', fallback: '/fallback',
    })
    const controller = new AbortController()
    const pending = PlayerFactory.create('cam1', document.createElement('video'), () => false, undefined, controller.signal)
    await flushPromises()
    expect(FakeSocket.instances).toHaveLength(1)
    controller.abort()

    await expect(pending).resolves.toBeNull()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('does not negotiate on a peer whose offer resolves after cancellation', async () => {
    let resolveOffer!: (offer: RTCSessionDescriptionInit) => void
    const offer = new Promise<RTCSessionDescriptionInit>(resolve => { resolveOffer = resolve })
    class SlowPeer extends FakePeer {
      createOffer = vi.fn().mockReturnValue(offer)
    }
    vi.stubGlobal('RTCPeerConnection', SlowPeer)
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    const peer = FakePeer.instances[0]
    player.destroy()
    await expect(pending).resolves.toBe(false)
    resolveOffer({ type: 'offer', sdp: 'late-offer' })
    await flushPromises()

    expect(peer?.setLocalDescription).not.toHaveBeenCalled()
    expect(FakeSocket.instances[0]?.send).not.toHaveBeenCalled()
  })
})
