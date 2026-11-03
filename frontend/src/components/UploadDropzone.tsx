import { useCallback, useRef, useState } from 'react'
import { Upload } from 'lucide-react'

interface UploadDropzoneProps {
  file: File | null
  onFileSelect: (file: File) => void
  acceptedTypes?: string
  maxSize?: number
  hint?: string
}

export function UploadDropzone({
  file,
  onFileSelect,
  acceptedTypes = 'image/*',
  maxSize = 10 * 1024 * 1024,
  hint,
}: UploadDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = useCallback(
    (candidate: File) => {
      setLocalError(null)
      if (candidate.size > maxSize) {
        setLocalError(
          `File is ${(candidate.size / 1024 / 1024).toFixed(1)} MB; limit is ` +
            `${(maxSize / 1024 / 1024).toFixed(0)} MB.`,
        )
        return
      }
      onFileSelect(candidate)
    },
    [maxSize, onFileSelect],
  )

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      setIsDragOver(false)
      const dropped = event.dataTransfer.files?.[0]
      if (dropped) accept(dropped)
    },
    [accept],
  )

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click()
        }}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        className={[
          'flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors',
          isDragOver
            ? 'border-brand-500 bg-brand-50'
            : 'border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50',
        ].join(' ')}
      >
        <Upload size={40} className="mb-3 text-slate-400" aria-hidden />
        {file ? (
          <>
            <p className="font-medium text-slate-900">{file.name}</p>
            <p className="text-sm text-slate-500">
              {(file.size / 1024 / 1024).toFixed(2)} MB — click to replace
            </p>
          </>
        ) : (
          <>
            <p className="font-medium text-slate-900">Drop a file here or click to browse</p>
            <p className="mt-1 text-sm text-slate-500">{hint ?? 'Images only'}</p>
            <p className="mt-1 text-xs text-slate-400">
              Max {(maxSize / 1024 / 1024).toFixed(0)} MB
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={acceptedTypes}
          className="hidden"
          onChange={(event) => {
            const selected = event.target.files?.[0]
            if (selected) accept(selected)
            event.target.value = ''
          }}
        />
      </div>
      {localError && (
        <p className="mt-2 text-sm text-red-600" role="alert">
          {localError}
        </p>
      )}
    </div>
  )
}
