import React, { startTransition, useEffect, useRef, useState } from 'react';
import { Search, Loader2, ChevronRight } from 'lucide-react';
import { GoogleGenAI, Type } from '@google/genai';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';

const SEARCH_MODEL = 'gemini-3-flash-preview';
const MOBILE_MEDIA_QUERY = '(max-width: 1023px)';

const normalizeEnvValue = (value?: string) => {
  const trimmed = value?.trim() ?? '';

  if (!trimmed || trimmed.startsWith('MY_')) {
    return '';
  }

  return trimmed;
};

const GEMINI_API_KEY =
  normalizeEnvValue(import.meta.env.GEMINI_API_KEY) ||
  normalizeEnvValue(import.meta.env.VITE_GEMINI_API_KEY);

const MISSING_API_KEY_MESSAGE =
  'Gemini API 키가 없어요. 프로젝트 루트의 .env.local 파일에 GEMINI_API_KEY 또는 VITE_GEMINI_API_KEY를 넣어 주세요.';
const INVALID_API_KEY_MESSAGE =
  'Gemini API 키가 유효하지 않아요. .env.local의 키가 정확한지 확인하거나 Google AI Studio에서 새 키를 발급해 넣어 주세요.';

let aiInstance: GoogleGenAI | null = null;
const basicCache = new Map<string, MeaningResult>();
const detailCache = new Map<string, Syllable[]>();

const getAI = () => {
  if (!GEMINI_API_KEY) {
    throw new Error(MISSING_API_KEY_MESSAGE);
  }

  if (!aiInstance) {
    aiInstance = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
  }

  return aiInstance;
};

interface Meaning {
  meaning: string;
  example: string;
}

interface Syllable {
  char: string;
  isHanja: boolean;
  hanjaChar?: string;
  hanjaMeaning?: string;
  relatedWords?: string[];
}

interface MeaningResult {
  word: string;
  meanings: Meaning[];
}

interface DictionaryResult {
  word: string;
  meanings: Meaning[] | null;
  syllables: Syllable[] | null;
}

const MEANING_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    word: { type: Type.STRING },
    meanings: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          meaning: { type: Type.STRING },
          example: { type: Type.STRING },
        },
        required: ['meaning', 'example'],
      },
    },
  },
  required: ['word', 'meanings'],
};

const SYLLABLE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    syllables: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          char: { type: Type.STRING },
          isHanja: { type: Type.BOOLEAN },
          hanjaChar: { type: Type.STRING },
          hanjaMeaning: { type: Type.STRING },
          relatedWords: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
          },
        },
        required: ['char', 'isHanja'],
      },
    },
  },
  required: ['syllables'],
};

const normalizeKey = (value: string) => value.trim().toLowerCase();

const parseJsonResponse = <T,>(text?: string) => {
  const jsonText = (text ?? '{}')
    .replace(/```json/gi, '')
    .replace(/```/g, '')
    .trim();

  return JSON.parse(jsonText) as T;
};

const formatErrorMessage = (error: unknown) => {
  const message = error instanceof Error ? error.message : String(error ?? '');

  if (/API_KEY_INVALID|API key not valid/i.test(message)) {
    return INVALID_API_KEY_MESSAGE;
  }

  if (/GEMINI_API_KEY|VITE_GEMINI_API_KEY/.test(message)) {
    return MISSING_API_KEY_MESSAGE;
  }

  return `검색 중 문제가 생겼어요. ${message || '잠시 후 다시 시도해 주세요.'}`;
};

const createFallbackSyllables = (word: string): Syllable[] =>
  Array.from(word).map((char) => ({
    char,
    isHanja: false,
  }));

const sanitizeMeanings = (query: string, payload: MeaningResult): MeaningResult => {
  const meanings = Array.isArray(payload.meanings)
    ? payload.meanings
        .map((item) => ({
          meaning: item.meaning?.trim() ?? '',
          example: item.example?.trim() ?? '',
        }))
        .filter((item) => item.meaning && item.example)
        .slice(0, 2)
    : [];

  if (!meanings.length) {
    throw new Error('검색 결과를 찾지 못했어요.');
  }

  return {
    word: payload.word?.trim() || query,
    meanings,
  };
};

const sanitizeSyllables = (word: string, syllables?: Syllable[]): Syllable[] => {
  const chars = Array.from(word);

  if (!Array.isArray(syllables) || syllables.length !== chars.length) {
    return createFallbackSyllables(word);
  }

  return chars.map((char, index) => {
    const source = syllables[index];
    const matchesChar = source?.char === char;
    const isHanja = matchesChar ? Boolean(source.isHanja) : false;
    const relatedWords = isHanja
      ? (source.relatedWords ?? [])
          .map((item) => item.trim())
          .filter(Boolean)
          .slice(0, 4)
      : undefined;

    return {
      char,
      isHanja,
      hanjaChar: isHanja && source.hanjaChar?.trim() ? source.hanjaChar.trim() : undefined,
      hanjaMeaning:
        isHanja && source.hanjaMeaning?.trim() ? source.hanjaMeaning.trim() : undefined,
      relatedWords: relatedWords?.length ? relatedWords : undefined,
    };
  });
};

const getMobileMatch = () => {
  if (typeof window === 'undefined') {
    return false;
  }

  return window.matchMedia(MOBILE_MEDIA_QUERY).matches;
};

const useIsMobileLayout = () => {
  const [isMobile, setIsMobile] = useState(getMobileMatch);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const mediaQuery = window.matchMedia(MOBILE_MEDIA_QUERY);
    setIsMobile(mediaQuery.matches);

    const handleChange = (event: MediaQueryListEvent) => {
      setIsMobile(event.matches);
    };

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }

    mediaQuery.addListener(handleChange);
    return () => mediaQuery.removeListener(handleChange);
  }, []);

  return isMobile;
};

type LoadingBuddyTone = 'insight' | 'meaning';

interface LoadingBuddyProps {
  word: string;
  title?: string;
  description?: string;
  badgeText?: string;
  ariaLabel?: string;
  tone?: LoadingBuddyTone;
}

const loadingBuddyPalette: Record<
  LoadingBuddyTone,
  {
    background: string;
    border: string;
    badge: string;
    title: string;
    description: string;
    tile: string;
    iconWrap: string;
    icon: string;
    glowPrimary: string;
    glowSecondary: string;
    dot: string;
  }
> = {
  insight: {
    background: 'bg-gradient-to-br from-white via-[#eef4ff] to-[#ffe8f1]',
    border: 'border-[#9fc0ff]',
    badge: 'border-[#9fc0ff] bg-white/90 text-[#245cff]',
    title: 'text-[#17366b]',
    description: 'text-[#4e6891]',
    tile: 'border-[#9fc0ff] bg-white text-[#245cff] shadow-[0_12px_22px_rgba(47,99,255,0.18)]',
    iconWrap: 'border-[#ffc7da] bg-white/95',
    icon: 'text-[#d9386a]',
    glowPrimary: 'bg-[#c7dbff]',
    glowSecondary: 'bg-[#ffd7e6]',
    dot: 'bg-[#d9386a]',
  },
  meaning: {
    background: 'bg-gradient-to-br from-white via-[#fff0f6] to-[#eef4ff]',
    border: 'border-[#ffc7da]',
    badge: 'border-[#ffc7da] bg-white/90 text-[#d9386a]',
    title: 'text-[#17366b]',
    description: 'text-[#4e6891]',
    tile: 'border-[#9fc0ff] bg-white text-[#245cff] shadow-[0_12px_22px_rgba(47,99,255,0.18)]',
    iconWrap: 'border-[#9fc0ff] bg-white/95',
    icon: 'text-[#245cff]',
    glowPrimary: 'bg-[#ffd7e6]',
    glowSecondary: 'bg-[#c7dbff]',
    dot: 'bg-[#245cff]',
  },
};

function LoadingBuddy({
  word,
  title,
  description,
  badgeText,
  ariaLabel = '불러오는 중',
  tone = 'insight',
}: LoadingBuddyProps) {
  const palette = loadingBuddyPalette[tone];
  const chars = Array.from(word.trim()).slice(0, 4);
  const displayChars = chars.length ? chars : ['?'];
  const hasTextBlock = Boolean(title || description);

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={ariaLabel}
      className={`relative overflow-hidden rounded-[2rem] border-2 border-dashed ${palette.background} ${palette.border} p-6`}
    >
      <motion.div
        aria-hidden
        className={`absolute -right-6 -top-6 h-24 w-24 rounded-full ${palette.glowPrimary}`}
        animate={{ scale: [1, 1.16, 1], opacity: [0.25, 0.45, 0.25] }}
        transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        aria-hidden
        className={`absolute -left-8 bottom-0 h-16 w-16 rounded-full ${palette.glowSecondary}`}
        animate={{ y: [0, -10, 0], opacity: [0.14, 0.28, 0.14] }}
        transition={{ duration: 2.9, repeat: Infinity, ease: 'easeInOut' }}
      />

      <div className="relative flex flex-col items-center gap-5 text-center">
        {badgeText && (
          <motion.div
            className={`inline-flex items-center rounded-full border px-4 py-2 text-sm font-black ${palette.badge}`}
            animate={{ y: [0, -4, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
          >
            {badgeText}
          </motion.div>
        )}

        <div className="flex flex-wrap items-end justify-center gap-3">
          {displayChars.map((char, index) => (
            <motion.div
              key={`${char}-${index}`}
              className={`flex h-20 w-20 items-center justify-center rounded-[1.6rem] border-2 text-4xl font-black shadow-lg ${palette.tile}`}
              animate={{ y: [0, -12, 0], rotate: [-4, 4, -4], scale: [1, 1.04, 1] }}
              transition={{
                duration: 1.7,
                repeat: Infinity,
                ease: 'easeInOut',
                delay: index * 0.14,
              }}
            >
              {char}
            </motion.div>
          ))}
        </div>

        <div className="flex items-center gap-2">
          {Array.from({ length: 3 }).map((_, index) => (
            <motion.span
              key={index}
              className={`h-3 w-3 rounded-full ${palette.dot}`}
              animate={{ y: [0, -6, 0], opacity: [0.35, 1, 0.35] }}
              transition={{
                duration: 1.1,
                repeat: Infinity,
                ease: 'easeInOut',
                delay: index * 0.12,
              }}
            />
          ))}
        </div>

        {hasTextBlock && (
          <div className="space-y-2">
            {title && <p className={`text-2xl font-black ${palette.title}`}>{title}</p>}
            {description && <p className={`text-base font-bold ${palette.description}`}>{description}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

interface RelatedWordChipProps {
  word: string;
  highlightChar: string;
  index: number;
  compact?: boolean;
}

function RelatedWordChip({
  word,
  highlightChar,
  index,
  compact = false,
}: RelatedWordChipProps) {
  const chars = Array.from(word);
  const hasHighlight = chars.some((char) => char === highlightChar);
  const tiltClass = compact ? '' : index % 2 === 0 ? '-rotate-[1.2deg]' : 'rotate-[1.2deg]';

  return (
    <span
      className={`inline-flex items-center gap-2 border-[3px] border-[#9fc0ff] bg-white font-black text-slate-700 shadow-[0_14px_28px_rgba(47,99,255,0.14)] ${tiltClass} ${
        compact ? 'rounded-[1.75rem] px-3 py-2 text-xl' : 'rounded-[2rem] px-4 py-3 text-3xl'
      }`}
      aria-label={hasHighlight ? `${word}에서 ${highlightChar}가 들어간 자리` : word}
    >
      {chars.map((char, index) => {
        const isMatch = char === highlightChar;

        return isMatch ? (
          <motion.span
            key={`${word}-${index}`}
            initial={{ scale: 0.88, y: 4 }}
            animate={{ scale: 1, y: 0 }}
            transition={{ duration: 0.24, delay: index * 0.05 }}
            className={`inline-flex items-center justify-center rounded-full bg-[#e33f78] text-white ring-[3px] ring-[#ffd3e2] ${
              compact ? 'h-11 min-w-[2.75rem] px-2 text-2xl' : 'h-[3.25rem] min-w-[3.25rem] px-3 text-3xl'
            }`}
          >
            {char}
          </motion.span>
        ) : (
          <span key={`${word}-${index}`} className={compact ? 'text-2xl' : 'text-4xl'}>
            {char}
          </span>
        );
      })}
    </span>
  );
}

interface RelatedWordShowcaseProps {
  words?: string[];
  highlightChar: string;
  compact?: boolean;
}

function RelatedWordShowcase({
  words,
  highlightChar,
  compact = false,
}: RelatedWordShowcaseProps) {
  if (!words?.length) {
    return null;
  }

  return (
    <div
      className={`rounded-[1.8rem] border-2 border-[#bfd3ff] bg-white/85 ${
        compact ? 'mt-4 p-4' : 'mt-5 p-5'
      }`}
    >
      <div className={`flex flex-wrap ${compact ? 'gap-2.5' : 'gap-3'}`}>
        {words.map((word, index) => (
          <RelatedWordChip
            key={`${word}-${index}`}
            word={word}
            highlightChar={highlightChar}
            index={index}
            compact={compact}
          />
        ))}
      </div>
    </div>
  );
}

const fetchMeaningResult = async (query: string): Promise<MeaningResult> => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: SEARCH_MODEL,
    contents: `너는 초등학교 3학년도 이해할 수 있게 낱말을 설명하는 도우미야.
반드시 JSON만 반환해.

규칙:
- word는 검색어 그대로 적어.
- meanings는 1~2개만 작성해.
- meaning은 초3이 읽어도 바로 뜻을 알 수 있게 아주 쉽게 써.
- 어려운 말, 사전 말투, 돌려 말하기를 쓰지 마.
- meaning은 한 문장으로, 가능하면 20자 안팎으로 짧게 써.
- example은 학교, 집, 친구, 놀이처럼 아이에게 익숙한 상황으로 써.
- example도 너무 길지 않게 한 문장으로만 써.
- example에는 검색어를 **굵게** 표시해.
- 뜻이 어려운 낱말이면 더 쉬운 말로 바꿔 풀어 써.

좋은 예:
- meaning: "여럿 가운데 하나를 고르는 것"
- meaning: "빛깔이 푸른 색"

검색어: ${query}`,
    config: {
      responseMimeType: 'application/json',
      responseSchema: MEANING_SCHEMA,
    },
  });

  return sanitizeMeanings(query, parseJsonResponse<MeaningResult>(response.text));
};

const fetchSyllableDetails = async (query: string): Promise<Syllable[]> => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: SEARCH_MODEL,
    contents: `아래 낱말을 초등학교 3학년도 이해할 수 있게 글자별로 알려 줘.
반드시 JSON만 반환해.

규칙:
- syllables는 검색어의 각 글자 순서대로 작성해.
- 한자어인 글자만 isHanja를 true로 하고 hanjaChar, hanjaMeaning, relatedWords를 채워.
- 고유어, 외래어, 추정이 어려운 글자는 isHanja를 false로 둬.
- hanjaMeaning은 초3도 알 만한 아주 쉬운 말 1개나 짧은 말로 써.
- hanjaMeaning에는 어려운 한자말이나 설명투를 쓰지 마.
- relatedWords는 그 한자가 실제로 들어가는 쉬운 낱말 2~3개만 넣어.
- relatedWords는 교과서나 일상에서 자주 볼 만한 낱말로 골라.
- 헷갈리면 억지로 맞추지 말고 isHanja를 false로 둬.

좋은 예:
- hanjaMeaning: "푸르다"
- hanjaMeaning: "끝"

검색어: ${query}`,
    config: {
      responseMimeType: 'application/json',
      responseSchema: SYLLABLE_SCHEMA,
    },
  });

  const payload = parseJsonResponse<{ syllables?: Syllable[] }>(response.text);
  return sanitizeSyllables(query, payload.syllables);
};

export default function App() {
  const isMobile = useIsMobileLayout();
  const hasApiKey = Boolean(GEMINI_API_KEY);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<DictionaryResult | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingMeanings, setIsLoadingMeanings] = useState(false);
  const [error, setError] = useState('');
  const [detailError, setDetailError] = useState('');
  const [meaningError, setMeaningError] = useState('');
  const [selectedSyllableIndex, setSelectedSyllableIndex] = useState<number | null>(null);
  const [revealedSyllableIndexes, setRevealedSyllableIndexes] = useState<Set<number>>(new Set());
  const [showSearchResult, setShowSearchResult] = useState(false);
  const activeSearchId = useRef(0);

  const loadSearchResult = async (word: string, cacheKey: string, searchId: number) => {
    const cachedBasic = basicCache.get(cacheKey) ?? null;

    setShowSearchResult(true);
    setMeaningError('');

    if (cachedBasic) {
      startTransition(() => {
        setResult((current) => {
          if (!current || normalizeKey(current.word) !== cacheKey) {
            return current;
          }

          return {
            ...current,
            word: cachedBasic.word,
            meanings: cachedBasic.meanings,
          };
        });
      });
      return;
    }

    setIsLoadingMeanings(true);

    try {
      const meaningResult = await fetchMeaningResult(word).then((value) => {
        basicCache.set(cacheKey, value);
        return value;
      });

      if (activeSearchId.current !== searchId) {
        return;
      }

      startTransition(() => {
        setResult((current) => {
          if (!current || normalizeKey(current.word) !== cacheKey) {
            return current;
          }

          return {
            ...current,
            word: meaningResult.word,
            meanings: meaningResult.meanings,
          };
        });
      });
    } catch (err) {
      if (activeSearchId.current !== searchId) {
        return;
      }

      console.error(err);
      setMeaningError(formatErrorMessage(err));
    } finally {
      if (activeSearchId.current === searchId) {
        setIsLoadingMeanings(false);
      }
    }
  };

  const handleSearch = async (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();

    if (!hasApiKey) {
      setError(MISSING_API_KEY_MESSAGE);
      return;
    }

    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const cacheKey = normalizeKey(trimmedQuery);
    const searchId = ++activeSearchId.current;
    const cachedBasic = basicCache.get(cacheKey) ?? null;
    const cachedDetails = detailCache.get(cacheKey) ?? null;

    setIsSearching(!cachedDetails);
    setIsLoadingMeanings(false);
    setError('');
    setDetailError('');
    setMeaningError('');
    setSelectedSyllableIndex(null);
    setRevealedSyllableIndexes(new Set());
    setShowSearchResult(false);

    startTransition(() => {
      setResult({
        word: cachedBasic?.word ?? trimmedQuery,
        meanings: cachedBasic?.meanings ?? null,
        syllables: cachedDetails,
      });
    });

    if (cachedDetails) {
      if (!cachedDetails.some((syllable) => syllable.isHanja)) {
        void loadSearchResult(cachedBasic?.word ?? trimmedQuery, cacheKey, searchId);
      }
      return;
    }

    try {
      const syllables = await fetchSyllableDetails(trimmedQuery).then((value) => {
        detailCache.set(cacheKey, value);
        return value;
      });

      if (activeSearchId.current !== searchId) {
        return;
      }

      startTransition(() => {
        setResult((current) => {
          if (!current || normalizeKey(current.word) !== cacheKey) {
            return {
              word: cachedBasic?.word ?? trimmedQuery,
              meanings: cachedBasic?.meanings ?? null,
              syllables,
            };
          }

          return {
            ...current,
            syllables,
          };
        });
      });

      if (!syllables.some((syllable) => syllable.isHanja)) {
        void loadSearchResult(cachedBasic?.word ?? trimmedQuery, cacheKey, searchId);
      }
    } catch (err) {
      if (activeSearchId.current !== searchId) {
        return;
      }

      console.error(err);
      setDetailError('글자별 분석을 불러오지 못했어요. 그래도 뜻풀이는 버튼으로 열 수 있어요.');
    } finally {
      if (activeSearchId.current === searchId) {
        setIsSearching(false);
      }
    }
  };

  const handleRevealSearchResult = async () => {
    if (!result) return;

    const cacheKey = normalizeKey(result.word);
    const currentSearchId = activeSearchId.current;
    await loadSearchResult(result.word, cacheKey, currentSearchId);
  };

  const handleSyllableClick = (index: number, isHanja: boolean) => {
    if (!isHanja) return;

    setSelectedSyllableIndex((current) => {
      if (current === index) {
        return null;
      }

      return index;
    });
  };

  const handleRevealHint = () => {
    if (selectedSyllableIndex === null) return;

    setRevealedSyllableIndexes((current) => {
      const next = new Set(current);
      next.add(selectedSyllableIndex);
      return next;
    });
  };

  const hasHanja = result?.syllables?.some((syllable) => syllable.isHanja) ?? false;
  const selectedSyllable =
    selectedSyllableIndex !== null ? result?.syllables?.[selectedSyllableIndex] : null;
  const isSelectedSyllableRevealed =
    selectedSyllableIndex !== null && revealedSyllableIndexes.has(selectedSyllableIndex);
  const showWordInsightPanel = Boolean(result) && (hasHanja || Boolean(detailError));
  const showSearchResultPanel = Boolean(result) && showSearchResult;
  const showStandaloneLoadingPanel =
    Boolean(result) && isSearching && !result?.syllables && !showWordInsightPanel && !showSearchResultPanel;
  const showRevealCard = false;
  const isInitial = !result && !isSearching;
  const mobileStatusTags: string[] = [];

  if (result) {
    mobileStatusTags.push(isSearching ? '글자 분석 중' : '검색 완료');

    if (hasHanja) {
      mobileStatusTags.push('한자 풀이 가능');
    } else if (result.syllables) {
      mobileStatusTags.push('뜻 중심 보기');
    }

    if (showSearchResultPanel) {
      mobileStatusTags.push(isLoadingMeanings && !result.meanings ? '뜻 불러오는 중' : '뜻 카드 열림');
    }
  }

  if (isMobile) {
    return (
      <div className="min-h-[100dvh] bg-[radial-gradient(circle_at_top,#eef4ff_0%,#f8fafc_42%,#fff5f9_100%)] px-4 pb-[calc(1.25rem+env(safe-area-inset-bottom))]">
        <div className="mx-auto max-w-xl">
          <div className="sticky top-0 z-20 -mx-4 border-b border-white/70 bg-[#f8fafc]/90 px-4 pb-4 pt-[calc(1rem+env(safe-area-inset-top))] backdrop-blur-xl">
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-3 px-1">
                <div>
                  <p className="text-xs font-black uppercase tracking-[0.22em] text-[#4e6891]">
                    Mobile Search
                  </p>
                  <h1 className="mt-2 text-2xl font-black text-[#17366b]">낱말 사전</h1>
                </div>

                {result && (
                  <div className="rounded-full border border-[#9fc0ff] bg-white/90 px-4 py-2 text-sm font-black text-[#245cff] shadow-sm">
                    {result.word}
                  </div>
                )}
              </div>

              <motion.form
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                onSubmit={handleSearch}
                className="rounded-[1.75rem] border border-white/80 bg-white/92 p-3 shadow-[0_18px_45px_rgba(15,23,42,0.12)] backdrop-blur-xl"
              >
                <label
                  htmlFor="mobile-word-search"
                  className="block px-1 text-xs font-black uppercase tracking-[0.22em] text-[#4e6891]"
                >
                  찾고 싶은 낱말
                </label>

                <div className="mt-3 flex items-center gap-3">
                  <input
                    id="mobile-word-search"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="예: 선택, 운동, 가족"
                    className="min-w-0 flex-1 rounded-[1.3rem] border-2 border-slate-200 bg-[#f8fbff] px-4 py-4 text-lg font-bold text-slate-800 outline-none transition focus:border-[#245cff] focus:ring-4 focus:ring-[#dce7ff] placeholder:text-slate-400"
                    disabled={isSearching}
                  />
                  <button
                    type="submit"
                    disabled={isSearching || !query.trim() || !hasApiKey}
                    className="flex h-14 w-14 shrink-0 items-center justify-center rounded-[1.25rem] border-2 border-[#ffc7da] bg-[#fff0f6] text-[#d9386a] shadow-[0_12px_24px_rgba(217,56,106,0.16)] transition hover:-translate-y-0.5 hover:bg-[#ffe4ef] disabled:opacity-50"
                  >
                    {isSearching ? (
                      <Loader2 className="h-6 w-6 animate-spin text-[#d9386a]" />
                    ) : (
                      <Search className="h-6 w-6" />
                    )}
                  </button>
                </div>
              </motion.form>

              {!hasApiKey && (
                <div className="w-full rounded-[1.6rem] border border-[#ffc7da] bg-[#fff1f6] px-4 py-4 text-sm font-bold text-[#9b355f]">
                  검색을 하려면 `.env.local`에 `GEMINI_API_KEY="..."` 또는
                  `VITE_GEMINI_API_KEY="..."`를 넣어 주세요.
                </div>
              )}

              {error && (
                <motion.div
                  key="mobile-error"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="rounded-[1.6rem] bg-red-50 px-4 py-3 text-center text-sm font-bold text-red-600"
                >
                  {error}
                </motion.div>
              )}

              {mobileStatusTags.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {mobileStatusTags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-slate-200 bg-white/90 px-3 py-1.5 text-xs font-black text-slate-600 shadow-sm"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4 pt-4">
            {isInitial && !error && (
              <motion.section
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                className="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-[#17366b] via-[#245cff] to-[#d9386a] p-6 text-white shadow-[0_30px_70px_rgba(36,92,255,0.28)]"
              >
                <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-white/12 blur-2xl" />
                <div className="absolute -bottom-12 left-0 h-28 w-28 rounded-full bg-[#ffd1e1]/25 blur-2xl" />

                <div className="relative">
                  <div className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-black uppercase tracking-[0.2em] text-white/80">
                    <Search className="h-3.5 w-3.5" />
                    Mobile Layout
                  </div>

                  <h2 className="mt-4 text-3xl font-black leading-tight">
                    손안에서 바로 보는
                    <br />
                    낱말 탐험
                  </h2>
                  <p className="mt-3 text-sm leading-6 text-white/84">
                    모바일에서는 검색, 글자 풀이, 뜻 보기를 세로 카드 흐름으로 나눠서 보여줘요.
                  </p>

                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-[1.4rem] border border-white/15 bg-white/10 p-4 backdrop-blur">
                      <p className="text-xs font-black uppercase tracking-[0.18em] text-white/70">탐색</p>
                      <p className="mt-2 text-lg font-black">글자 단위 풀이</p>
                    </div>
                    <div className="rounded-[1.4rem] border border-white/15 bg-white/10 p-4 backdrop-blur">
                      <p className="text-xs font-black uppercase tracking-[0.18em] text-white/70">예문</p>
                      <p className="mt-2 text-lg font-black">쉬운 뜻과 문장</p>
                    </div>
                  </div>
                </div>
              </motion.section>
            )}

            <AnimatePresence mode="wait">
              {result && !error && (
                <motion.div
                  key={`mobile-${result.word}-${showSearchResultPanel}-${showWordInsightPanel}`}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -16 }}
                  className="space-y-4"
                >
                  {showWordInsightPanel && (
                    <motion.section
                      layout
                      className="relative isolate w-full rounded-[2rem] border-2 border-slate-200/60 bg-white p-5 shadow-sm"
                    >
                      <div className="mb-5 space-y-2">
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div className="space-y-2">
                            <span className="inline-flex rounded-full border border-[#9fc0ff] bg-[#eef4ff] px-3 py-1 text-xs font-black uppercase tracking-[0.18em] text-[#245cff]">
                              글자 풀이
                            </span>
                            <h2 className="text-2xl font-black text-[#17366b]">낱말 속 글자를 살펴봐요</h2>
                          </div>

                          {!showSearchResult && !isSearching && (
                            <button
                              onClick={handleRevealSearchResult}
                              disabled={isLoadingMeanings}
                              className="inline-flex w-full items-center justify-center gap-2 rounded-full border-2 border-[#ffc7da] bg-[#fff0f6] px-5 py-3 text-base font-black text-[#d9386a] transition-all shadow-sm shadow-[0_10px_20px_rgba(217,56,106,0.16)] hover:-translate-y-0.5 hover:bg-[#ffe4ef] disabled:opacity-60"
                            >
                              {isLoadingMeanings ? (
                                <>
                                  <Loader2 className="h-5 w-5 animate-spin" />
                                  뜻 불러오는 중
                                </>
                              ) : (
                                <>
                                  뜻도 보기
                                  <ChevronRight className="h-5 w-5" />
                                </>
                              )}
                            </button>
                          )}
                        </div>

                        <p className="text-sm font-bold text-[#4e6891]">
                          눌러서 어떤 한자가 숨어 있는지 찾아보세요.
                        </p>
                      </div>

                      {isSearching && !result.syllables && (
                        <LoadingBuddy
                          word={result.word}
                          ariaLabel="글자 풀이를 준비하는 중"
                          tone="insight"
                        />
                      )}

                      {!isSearching && detailError && !result.syllables && (
                        <div className="rounded-[1.6rem] border border-red-200 bg-red-50 p-4 text-sm font-bold text-red-700">
                          {detailError}
                        </div>
                      )}

                      {hasHanja && result.syllables && (
                        <>
                          <div className="-mx-1 flex gap-3 overflow-x-auto px-1 pb-2">
                            {result.syllables.map((syllable, index) => (
                              <button
                                key={index}
                                onClick={() => handleSyllableClick(index, syllable.isHanja)}
                                disabled={!syllable.isHanja}
                                className={`shrink-0 rounded-[1.35rem] font-black transition-all ${
                                  !syllable.isHanja
                                    ? 'h-[4.75rem] min-w-[4.75rem] bg-slate-100 text-3xl text-slate-400'
                                    : selectedSyllableIndex === index
                                      ? 'h-[4.75rem] min-w-[4.75rem] scale-105 bg-gradient-to-br from-[#2f63ff] to-[#245cff] text-3xl text-white shadow-[0_16px_30px_rgba(47,99,255,0.28)] ring-4 ring-[#dce7ff]'
                                      : 'h-[4.75rem] min-w-[4.75rem] border-2 border-[#9fc0ff] bg-gradient-to-b from-white to-[#eef4ff] text-3xl text-[#245cff] shadow-[0_12px_24px_rgba(47,99,255,0.16)]'
                                }`}
                              >
                                {syllable.char}
                              </button>
                            ))}
                          </div>

                          {!selectedSyllable && (
                            <div className="rounded-[1.6rem] border-2 border-dashed border-[#9fc0ff] bg-[#f8fbff] p-4 text-center text-sm font-bold text-[#4e6891]">
                              글자를 눌러 보면 어떤 한자가 들어 있는지 볼 수 있어요.
                            </div>
                          )}

                          <AnimatePresence mode="wait">
                            {selectedSyllable && (
                              <motion.div
                                key={`mobile-${selectedSyllable.char}-${selectedSyllableIndex}`}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="mt-4 overflow-hidden rounded-[2rem] border-2 border-[#9fc0ff] bg-gradient-to-br from-white via-[#eef4ff] to-[#fff0f6] p-5 shadow-[0_20px_45px_rgba(47,99,255,0.12)]"
                              >
                                <div className="flex flex-wrap items-start justify-between gap-4">
                                  <div>
                                    <p className="text-xs font-black uppercase tracking-[0.22em] text-[#4e6891]">
                                      선택한 글자
                                    </p>
                                    <div className="mt-2 flex items-end gap-3">
                                      <span className="text-4xl font-black text-[#17366b]">
                                        {selectedSyllable.char}
                                      </span>
                                      {selectedSyllable.hanjaChar && (
                                        <span className="text-2xl font-black text-[#d9386a]">
                                          {selectedSyllable.hanjaChar}
                                        </span>
                                      )}
                                      <span className="pb-1 text-lg font-black text-[#17366b]">
                                        이 들어간 낱말
                                      </span>
                                    </div>
                                  </div>
                                </div>

                                <RelatedWordShowcase
                                  words={selectedSyllable.relatedWords}
                                  highlightChar={selectedSyllable.char}
                                  compact
                                />

                                <p className="mb-4 mt-5 text-lg font-black text-[#d9386a]">무슨 뜻일까요?</p>

                                {!isSelectedSyllableRevealed ? (
                                  <button
                                    onClick={handleRevealHint}
                                    className="flex w-full items-center justify-center gap-2 rounded-full border-2 border-white/70 bg-gradient-to-r from-[#ff6b93] to-[#d9386a] px-5 py-3 text-lg font-black text-white shadow-md shadow-[0_14px_28px_rgba(217,56,106,0.26)] transition-all hover:-translate-y-1"
                                  >
                                    정답 확인하기
                                    <ChevronRight className="h-5 w-5" />
                                  </button>
                                ) : (
                                  <motion.div
                                    initial={{ opacity: 0, scale: 0.9, y: 10 }}
                                    animate={{ opacity: 1, scale: 1, y: 0 }}
                                    className="rounded-[1.8rem] border-4 border-[#dce7ff] bg-[#245cff] px-5 py-5 text-center text-2xl font-black text-white shadow-lg"
                                  >
                                    {selectedSyllable.hanjaMeaning ?? '뜻 풀이가 아직 없어요.'}
                                  </motion.div>
                                )}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </>
                      )}
                    </motion.section>
                  )}

                  {showSearchResultPanel && (
                    <motion.section
                      layout
                      className="relative isolate w-full rounded-[2rem] border-2 border-slate-200/60 bg-white p-5 shadow-sm"
                    >
                      <div className="mb-5 space-y-2">
                        <span className="inline-flex rounded-full border border-[#ffc7da] bg-[#fff0f6] px-3 py-1 text-xs font-black uppercase tracking-[0.18em] text-[#d9386a]">
                          뜻 보기
                        </span>
                        <h2 className="text-3xl font-black text-[#17366b]">{result.word}</h2>
                        <p className="text-sm font-bold text-slate-500">
                          쉬운 설명과 예문을 한 번에 살펴봐요.
                        </p>
                      </div>

                      {meaningError && (
                        <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-bold text-red-600">
                          {meaningError}
                        </div>
                      )}

                      {isLoadingMeanings && !result.meanings && (
                        <div className="space-y-4">
                          <LoadingBuddy word={result.word} ariaLabel="뜻풀이를 불러오는 중" tone="meaning" />

                          <div className="space-y-3 animate-pulse">
                            {Array.from({ length: 2 }).map((_, index) => (
                              <div
                                key={index}
                                className="h-24 rounded-[1.5rem] border-2 border-slate-200 bg-slate-50"
                              />
                            ))}
                          </div>
                        </div>
                      )}

                      {result.meanings && (
                        <div className="space-y-4">
                          {result.meanings.map((item, index) => (
                            <div
                              key={index}
                              className={`overflow-hidden rounded-[1.7rem] border-2 border-slate-200 bg-slate-50 ${
                                index % 2 === 0 ? '-rotate-[0.15deg]' : 'rotate-[0.15deg]'
                              }`}
                            >
                              <div className="flex items-start gap-3 border-b-2 border-slate-100 bg-white p-4">
                                <span
                                  className={`flex h-9 w-9 items-center justify-center rounded-[0.9rem] bg-gradient-to-br from-[#ff6b93] to-[#d9386a] text-base font-black text-white shadow-md shadow-[0_10px_20px_rgba(217,56,106,0.2)] ${
                                    index % 2 === 0 ? '-rotate-[3deg]' : 'rotate-[3deg]'
                                  }`}
                                >
                                  {index + 1}
                                </span>
                                <p className="pt-0.5 text-lg font-bold leading-snug text-[#214c88]">
                                  {item.meaning}
                                </p>
                              </div>

                              <div className="flex items-start gap-3 bg-slate-50 p-4">
                                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#fff0f6] text-lg text-[#d9386a] shadow-sm">
                                  예
                                </span>
                                <div className="markdown-body-inline break-keep pt-0.5 text-base leading-snug text-slate-600">
                                  <Markdown>{item.example}</Markdown>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </motion.section>
                  )}

                  {showStandaloneLoadingPanel && (
                    <motion.section
                      layout
                      className="relative isolate w-full rounded-[2rem] border-2 border-slate-200/60 bg-white p-5 shadow-sm"
                    >
                      <LoadingBuddy word={result.word} ariaLabel="낱말을 살피는 중" tone="meaning" />
                    </motion.section>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`h-screen bg-[#f8fafc] flex flex-col items-center py-6 px-6 font-sans overflow-hidden transition-all duration-700 ${
        isInitial ? 'justify-center' : 'justify-start'
      }`}
    >
      <div
        className={`w-full max-w-[95vw] 2xl:max-w-[1600px] flex flex-col gap-6 min-h-0 ${
          isInitial ? '' : 'h-full'
        }`}
      >
        <motion.form
          layout
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, type: 'spring', bounce: 0.2 }}
          onSubmit={handleSearch}
          className={`relative mx-auto w-full shrink-0 transition-all duration-700 ${
            isInitial ? 'max-w-3xl' : 'max-w-4xl'
          }`}
        >
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="궁금한 낱말을 적어 보세요"
            className={`w-full rounded-full border-4 border-slate-200 focus:border-[#245cff] focus:ring-4 focus:ring-[#dce7ff] outline-none transition-all duration-700 bg-white font-bold text-slate-800 placeholder:text-slate-400 ${
              isInitial ? 'pl-10 pr-24 py-6 text-4xl shadow-2xl' : 'pl-8 pr-20 py-4 text-2xl shadow-md'
            }`}
            disabled={isSearching}
          />
          <button
            type="submit"
            disabled={isSearching || !query.trim() || !hasApiKey}
            className={`absolute top-1/2 -translate-y-1/2 text-[#d9386a] hover:text-[#c92f60] disabled:opacity-50 transition-all rounded-[1.5rem] border-2 border-[#ffc7da] bg-[#fff0f6] shadow-sm shadow-[0_10px_20px_rgba(217,56,106,0.16)] hover:-rotate-6 hover:bg-[#ffe4ef] ${
              isInitial ? 'right-4 p-4' : 'right-3 p-3'
            }`}
          >
            {isSearching ? (
              <Loader2 className={`animate-spin text-[#d9386a] ${isInitial ? 'w-10 h-10' : 'w-8 h-8'}`} />
            ) : (
              <Search className={`transition-all duration-700 ${isInitial ? 'w-10 h-10' : 'w-8 h-8'}`} />
            )}
          </button>
        </motion.form>

        {!hasApiKey && (
          <div className="max-w-4xl mx-auto w-full rounded-[2rem] border border-[#ffc7da] bg-[#fff1f6] px-6 py-5 text-[#9b355f] font-bold text-lg shrink-0">
            검색을 쓰려면 `.env.local`에 `GEMINI_API_KEY="..."` 또는 `VITE_GEMINI_API_KEY="..."`를 넣어야 해요.
          </div>
        )}

        <AnimatePresence mode="wait">
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 bg-red-50 text-red-600 rounded-2xl text-center font-bold text-xl max-w-4xl mx-auto w-full shrink-0"
            >
              {error}
            </motion.div>
          )}

          {result && !error && (
            <motion.div
              layout
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`mx-auto w-full flex-1 min-h-0 pb-2 ${
                showWordInsightPanel && showSearchResultPanel
                  ? 'grid gap-6 lg:grid-cols-[minmax(320px,0.9fr)_minmax(0,1.1fr)]'
                  : 'max-w-4xl'
              }`}
            >
              {showWordInsightPanel && (
                <motion.div
                  layout
                  className="order-1 relative isolate w-full p-8 bg-white rounded-[2.2rem] shadow-sm border-2 border-slate-200/60 flex flex-col min-h-0 overflow-y-auto custom-scrollbar"
                >
                  <div className="flex flex-wrap items-center justify-end gap-4 mb-6 shrink-0">
                    {!showSearchResult && !isSearching && (
                      <button
                        onClick={handleRevealSearchResult}
                        disabled={isLoadingMeanings}
                        className="inline-flex items-center gap-2 rounded-full border-2 border-[#ffc7da] bg-[#fff0f6] px-6 py-3 text-xl font-black text-[#d9386a] transition-all shadow-sm shadow-[0_10px_20px_rgba(217,56,106,0.16)] hover:-translate-y-0.5 hover:bg-[#ffe4ef] disabled:opacity-60"
                      >
                        {isLoadingMeanings ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            뜻풀이 준비 중
                          </>
                        ) : (
                          <>
                            뜻풀이 보기
                            <ChevronRight className="w-5 h-5" />
                          </>
                        )}
                      </button>
                    )}
                  </div>

                  {isSearching && !result.syllables && (
                    <LoadingBuddy
                      word={result.word}
                      ariaLabel="글자별 뜻을 준비하는 중"
                      tone="insight"
                    />
                  )}

                  {!isSearching && detailError && !result.syllables && (
                    <div className="rounded-[2rem] border border-red-200 bg-red-50 p-6 text-red-700 font-bold text-lg">
                      {detailError}
                    </div>
                  )}

                  {hasHanja && result.syllables && (
                    <>
                      <div className="flex flex-wrap items-center gap-4 mb-6 justify-center shrink-0">
                        {result.syllables.map((syllable, index) => (
                          <React.Fragment key={index}>
                            <button
                              onClick={() => handleSyllableClick(index, syllable.isHanja)}
                              disabled={!syllable.isHanja}
                              className={`w-24 h-24 text-5xl font-black rounded-[1.5rem] flex items-center justify-center transition-all ${
                                !syllable.isHanja
                                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                  : selectedSyllableIndex === index
                                    ? 'rotate-[-2deg] bg-gradient-to-br from-[#2f63ff] to-[#245cff] text-white shadow-lg shadow-[0_16px_30px_rgba(47,99,255,0.28)] scale-110 ring-4 ring-[#dce7ff]'
                                    : `${index % 2 === 0 ? '-rotate-[2deg]' : 'rotate-[2deg]'} bg-gradient-to-b from-white to-[#eef4ff] text-[#245cff] hover:bg-[#dce7ff] hover:scale-105 cursor-pointer shadow-[0_12px_24px_rgba(47,99,255,0.16)] border-2 border-[#9fc0ff]`
                              }`}
                            >
                              {syllable.char}
                            </button>
                            {index < result.syllables.length - 1 && (
                              <span className="flex h-12 w-12 items-center justify-center rounded-full border border-[#ffc7da] bg-white text-[#ffb8cf] font-black text-4xl shadow-sm">
                                +
                              </span>
                            )}
                          </React.Fragment>
                        ))}
                      </div>

                      {!selectedSyllable && (
                        <div className="rounded-[2rem] border-2 border-dashed border-[#9fc0ff] bg-[#f8fbff] p-6 text-center text-[#4e6891] text-lg font-bold">
                          파란 글자를 눌러서 어떤 한자가 들어 있는지 알아보세요.
                        </div>
                      )}

                      <AnimatePresence mode="wait">
                        {selectedSyllable && (
                          <motion.div
                            key={`${selectedSyllable.char}-${selectedSyllableIndex}`}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="mt-6 relative isolate overflow-hidden rounded-[2.2rem] border-2 border-[#9fc0ff] bg-gradient-to-br from-white via-[#eef4ff] to-[#fff0f6] p-6 shadow-[0_20px_45px_rgba(47,99,255,0.12)]"
                          >
                            <div className="flex flex-wrap items-start justify-between gap-4">
                              <div>
                                <div className="flex items-end gap-3">
                                  <span className="text-5xl font-black text-[#17366b]">
                                    {selectedSyllable.char}
                                  </span>
                                  {selectedSyllable.hanjaChar && (
                                    <span className="text-3xl font-black text-[#d9386a]">
                                      {selectedSyllable.hanjaChar}
                                    </span>
                                  )}
                                  <span className="pb-1 text-2xl font-black text-[#17366b]">
                                    이 들어간 낱말
                                  </span>
                                </div>
                              </div>
                            </div>

                            <RelatedWordShowcase
                              words={selectedSyllable.relatedWords}
                              highlightChar={selectedSyllable.char}
                            />

                            <p className="mt-6 mb-6 text-2xl font-bold text-[#d9386a]">무슨 뜻일까요?</p>
                            {!isSelectedSyllableRevealed ? (
                              <button
                                onClick={handleRevealHint}
                                className="px-8 py-4 bg-gradient-to-r from-[#ff6b93] to-[#d9386a] hover:from-[#ff7aa1] hover:to-[#c92f60] text-white font-black text-2xl rounded-full border-2 border-white/70 transition-all shadow-md shadow-[0_14px_28px_rgba(217,56,106,0.26)] hover:shadow-lg flex items-center gap-2 transform hover:-translate-y-1"
                              >
                                정답 확인하기 <ChevronRight className="w-6 h-6" />
                              </button>
                            ) : (
                              <motion.div
                                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                className="px-8 py-6 bg-[#245cff] text-white font-black text-3xl rounded-[2rem] shadow-lg border-4 border-[#dce7ff] text-center"
                              >
                                {selectedSyllable.hanjaMeaning ?? '뜻 풀이가 아직 없어요.'}
                              </motion.div>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </>
                  )}
                </motion.div>
              )}

              {showRevealCard && (
                <motion.div
                  layout
                  className="order-1 w-full p-8 bg-white rounded-[2rem] shadow-sm border-2 border-slate-200/60 flex flex-col items-center justify-center text-center gap-6 min-h-0"
                >
                  <div className="space-y-3">
                    <p className="text-sm font-bold uppercase tracking-[0.2em] text-slate-400">
                      뜻풀이
                    </p>
                    <h2 className="text-3xl font-black text-[#1f3d63]">{result.word}</h2>
                    <p className="text-lg font-bold text-slate-500">
                      이 낱말은 따로 보여 줄 글자별 뜻이 없어서 뜻풀이를 바로 볼 수 있어요.
                    </p>
                  </div>
                  <button
                    onClick={handleRevealSearchResult}
                    disabled={isLoadingMeanings}
                    className="inline-flex items-center gap-2 rounded-full border border-[#ffc7da] bg-[#fff0f6] px-8 py-4 text-2xl font-black text-[#d9386a] transition-colors hover:bg-[#ffe4ef] disabled:opacity-60"
                  >
                    {isLoadingMeanings ? (
                      <>
                        <Loader2 className="w-6 h-6 animate-spin" />
                        뜻풀이 준비 중
                      </>
                    ) : (
                      <>
                        뜻풀이 보기
                        <ChevronRight className="w-6 h-6" />
                      </>
                    )}
                  </button>
                </motion.div>
              )}

              {showStandaloneLoadingPanel && (
                <motion.div
                  layout
                  className="order-1 relative isolate w-full p-8 bg-white rounded-[2.2rem] shadow-sm border-2 border-slate-200/60 flex flex-col justify-center min-h-0"
                >
                  <LoadingBuddy
                    word={result.word}
                    ariaLabel="낱말을 살피는 중"
                    tone="meaning"
                  />
                </motion.div>
              )}

              {showSearchResultPanel && (
                <motion.div
                  layout
                  className="order-2 relative isolate w-full p-8 bg-white rounded-[2.2rem] shadow-sm border-2 border-slate-200/60 flex flex-col min-h-0 overflow-y-auto custom-scrollbar"
                >
                  <div className="flex flex-wrap items-start justify-between gap-4 mb-6 shrink-0">
                    <div>
                      <h1 className="text-4xl font-black text-[#17366b] -rotate-[1deg] origin-left inline-block">
                        {result.word}
                      </h1>
                    </div>

                    <div className="flex flex-wrap gap-2" />
                  </div>

                  {meaningError && (
                    <div className="mb-6 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-red-600 font-bold shrink-0">
                      {meaningError}
                    </div>
                  )}

                  {isLoadingMeanings && !result.meanings && (
                    <div className="space-y-6">
                      <LoadingBuddy
                        word={result.word}
                        ariaLabel="뜻풀이를 불러오는 중"
                        tone="meaning"
                      />

                      <div className="space-y-4 animate-pulse">
                        {Array.from({ length: 2 }).map((_, index) => (
                          <div
                            key={index}
                            className="h-32 rounded-[1.5rem] border-2 border-slate-200 bg-slate-50"
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {result.meanings && (
                    <div className="space-y-6">
                      {result.meanings.map((item, index) => (
                        <div
                          key={index}
                          className={`bg-slate-50 rounded-[1.8rem] border-2 border-slate-200 overflow-hidden shrink-0 ${
                            index % 2 === 0 ? '-rotate-[0.35deg]' : 'rotate-[0.35deg]'
                          }`}
                        >
                          <div className="p-6 bg-white border-b-2 border-slate-100 flex gap-4 items-start">
                            <span
                              className={`flex-shrink-0 w-10 h-10 bg-gradient-to-br from-[#ff6b93] to-[#d9386a] text-white rounded-[1rem] flex items-center justify-center font-black text-xl shadow-md shadow-[0_10px_20px_rgba(217,56,106,0.2)] ${
                                index % 2 === 0 ? '-rotate-[4deg]' : 'rotate-[4deg]'
                              }`}
                            >
                              {index + 1}
                            </span>
                            <p className="text-[#214c88] font-bold text-2xl leading-snug pt-1">
                              {item.meaning}
                            </p>
                          </div>

                          <div className="p-6 bg-slate-50 flex gap-4 items-start">
                            <span className="flex-shrink-0 w-10 h-10 bg-[#fff0f6] text-[#d9386a] rounded-xl flex items-center justify-center text-2xl shadow-sm">
                              예
                            </span>
                            <div className="text-slate-600 text-xl leading-snug pt-1 markdown-body-inline break-keep">
                              <Markdown>{item.example}</Markdown>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
