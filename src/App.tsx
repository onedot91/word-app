import React, { startTransition, useRef, useState } from 'react';
import { Search, Loader2, Lightbulb, ChevronRight } from 'lucide-react';
import { GoogleGenAI, Type } from '@google/genai';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';

const SEARCH_MODEL = 'gemini-3-flash-preview';

let aiInstance: GoogleGenAI | null = null;
const basicCache = new Map<string, MeaningResult>();
const detailCache = new Map<string, Syllable[]>();

const getAI = () => {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY가 설정되지 않았어요.');
  }

  if (!aiInstance) {
    aiInstance = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
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

interface DictionaryResult extends MeaningResult {
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

const fetchMeaningResult = async (query: string): Promise<MeaningResult> => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: SEARCH_MODEL,
    contents: `너는 초등학생용 낱말 설명 도우미야.
반드시 JSON만 반환해.

규칙:
- word는 검색어 그대로 적어.
- meanings는 1~2개만 작성해.
- meaning은 짧고 쉬운 한국어로 설명해.
- example은 검색어를 **굵게** 표시한 자연스러운 한 문장으로 써.

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
    contents: `아래 낱말을 글자별로 분석해서 JSON만 반환해.

규칙:
- syllables는 검색어의 각 글자 순서대로 작성해.
- 한자어인 글자만 isHanja를 true로 하고 hanjaChar, hanjaMeaning, relatedWords를 채워.
- 고유어, 외래어, 추정이 어려운 글자는 isHanja를 false로 둬.
- hanjaMeaning은 초등학생도 이해하기 쉬운 짧은 풀이로 써.
- relatedWords는 그 한자가 실제로 들어가는 쉬운 낱말 2~3개만 넣어.

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
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<DictionaryResult | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [error, setError] = useState('');
  const [detailError, setDetailError] = useState('');
  const [selectedSyllableIndex, setSelectedSyllableIndex] = useState<number | null>(null);
  const [showHint, setShowHint] = useState(false);
  const activeSearchId = useRef(0);

  const handleSearch = async (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const cacheKey = normalizeKey(trimmedQuery);
    const searchId = ++activeSearchId.current;
    const cachedBasic = basicCache.get(cacheKey);
    const cachedDetails = detailCache.get(cacheKey) ?? null;

    setIsSearching(true);
    setIsLoadingDetails(!cachedDetails);
    setError('');
    setDetailError('');
    setSelectedSyllableIndex(null);
    setShowHint(false);

    if (!cachedBasic) {
      startTransition(() => setResult(null));
    }

    const detailsPromise = cachedDetails
      ? null
      : fetchSyllableDetails(trimmedQuery)
          .then((syllables) => {
            detailCache.set(cacheKey, syllables);
            return syllables;
          })
          .catch((detailErr) => {
            console.error(detailErr);
            return null;
          });

    try {
      const basicResult =
        cachedBasic ??
        (await fetchMeaningResult(trimmedQuery).then((value) => {
          basicCache.set(cacheKey, value);
          return value;
        }));

      if (activeSearchId.current !== searchId) {
        return;
      }

      startTransition(() => {
        setResult({
          ...basicResult,
          syllables: detailCache.get(cacheKey) ?? cachedDetails,
        });
      });
      setIsSearching(false);

      if (!detailsPromise) {
        return;
      }

      const syllables = await detailsPromise;

      if (activeSearchId.current !== searchId) {
        return;
      }

      if (!syllables) {
        setDetailError('글자별 한자 분석은 아직 불러오지 못했어요. 뜻과 예문은 먼저 볼 수 있어요.');
        return;
      }

      startTransition(() => {
        setResult((current) => {
          if (!current) {
            return current;
          }

          return {
            ...current,
            syllables,
          };
        });
      });
    } catch (err: any) {
      if (activeSearchId.current !== searchId) {
        return;
      }

      console.error(err);
      setError(`검색 중 문제가 생겼어요. ${err?.message || '잠시 후 다시 시도해 주세요.'}`);
      startTransition(() => setResult(null));
    } finally {
      if (activeSearchId.current === searchId) {
        setIsSearching(false);
        setIsLoadingDetails(false);
      }
    }
  };

  const handleSyllableClick = (index: number, isHanja: boolean) => {
    if (!isHanja) return;

    setSelectedSyllableIndex((current) => {
      if (current === index) {
        return null;
      }

      setShowHint(false);
      return index;
    });
  };

  const hasHanja = result?.syllables?.some((syllable) => syllable.isHanja) ?? false;
  const selectedSyllable =
    selectedSyllableIndex !== null ? result?.syllables?.[selectedSyllableIndex] : null;
  const showSyllablePanel = Boolean(result) && (hasHanja || isLoadingDetails || detailError);
  const isInitial = !result && !isSearching;

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
            className={`w-full rounded-full border-4 border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 outline-none transition-all duration-700 bg-white font-bold text-slate-800 placeholder:text-slate-400 ${
              isInitial ? 'pl-10 pr-24 py-6 text-4xl shadow-2xl' : 'pl-8 pr-20 py-4 text-2xl shadow-md'
            }`}
            disabled={isSearching}
          />
          <button
            type="submit"
            disabled={isSearching || !query.trim()}
            className={`absolute top-1/2 -translate-y-1/2 text-slate-400 hover:text-blue-500 disabled:opacity-50 transition-colors rounded-full hover:bg-slate-50 ${
              isInitial ? 'right-4 p-4' : 'right-3 p-3'
            }`}
          >
            {isSearching ? (
              <Loader2 className={`animate-spin text-blue-500 ${isInitial ? 'w-10 h-10' : 'w-8 h-8'}`} />
            ) : (
              <Search className={`transition-all duration-700 ${isInitial ? 'w-10 h-10' : 'w-8 h-8'}`} />
            )}
          </button>
        </motion.form>

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
                showSyllablePanel
                  ? 'grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]'
                  : 'max-w-4xl'
              }`}
            >
              <motion.div
                layout
                className="w-full p-8 bg-white rounded-[2rem] shadow-sm border-2 border-slate-200/60 flex flex-col min-h-0 overflow-y-auto custom-scrollbar"
              >
                <div className="flex flex-wrap items-start justify-between gap-4 mb-6 shrink-0">
                  <div>
                    <p className="text-sm font-bold uppercase tracking-[0.2em] text-slate-400 mb-2">
                      Search Result
                    </p>
                    <h1 className="text-4xl font-black text-slate-900">{result.word}</h1>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {isLoadingDetails && (
                      <span className="inline-flex items-center gap-2 rounded-full bg-amber-50 px-4 py-2 text-sm font-bold text-amber-700 border border-amber-200">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        한자 풀이 불러오는 중
                      </span>
                    )}
                    {!isLoadingDetails && result.syllables && (
                      <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-4 py-2 text-sm font-bold text-emerald-700 border border-emerald-200">
                        분석 준비됨
                      </span>
                    )}
                  </div>
                </div>

                {detailError && (
                  <div className="mb-6 rounded-2xl border border-amber-200 bg-amber-50 px-5 py-4 text-amber-800 font-bold shrink-0">
                    {detailError}
                  </div>
                )}

                {result.syllables && !hasHanja && (
                  <div className="mb-6 p-5 bg-blue-50 text-blue-700 rounded-2xl font-bold text-xl flex items-center gap-3 shrink-0">
                    <Lightbulb className="w-6 h-6 shrink-0" />
                    이 낱말은 고유어나 외래어라서 글자별 한자 풀이가 없어요.
                  </div>
                )}

                <div className="space-y-6">
                  {result.meanings.map((item, index) => (
                    <div
                      key={index}
                      className="bg-slate-50 rounded-[1.5rem] border-2 border-slate-200 overflow-hidden shrink-0"
                    >
                      <div className="p-6 bg-white border-b-2 border-slate-100 flex gap-4 items-start">
                        <span className="flex-shrink-0 w-10 h-10 bg-slate-800 text-white rounded-xl flex items-center justify-center font-black text-xl shadow-md">
                          {index + 1}
                        </span>
                        <p className="text-slate-800 font-bold text-2xl leading-snug pt-1">
                          {item.meaning}
                        </p>
                      </div>

                      <div className="p-6 bg-slate-50 flex gap-4 items-start">
                        <span className="flex-shrink-0 w-10 h-10 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center text-2xl shadow-sm">
                          예
                        </span>
                        <div className="text-slate-600 text-xl leading-snug pt-1 markdown-body-inline break-keep">
                          <Markdown>{item.example}</Markdown>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>

              {showSyllablePanel && (
                <motion.div
                  layout
                  className="w-full p-8 bg-white rounded-[2rem] shadow-sm border-2 border-slate-200/60 flex flex-col min-h-0 overflow-y-auto custom-scrollbar"
                >
                  <div className="flex items-center gap-3 mb-6 shrink-0">
                    <span className="w-12 h-12 rounded-2xl bg-amber-100 text-amber-600 flex items-center justify-center">
                      <Lightbulb className="w-6 h-6" />
                    </span>
                    <div>
                      <p className="text-sm font-bold uppercase tracking-[0.2em] text-slate-400">
                        Word Insight
                      </p>
                      <h2 className="text-2xl font-black text-slate-900">글자별 이해</h2>
                    </div>
                  </div>

                  {isLoadingDetails && !result.syllables && (
                    <div className="space-y-6 animate-pulse">
                      <div className="flex justify-center gap-4">
                        {Array.from({ length: Math.max(Array.from(result.word).length, 2) }).map((_, index) => (
                          <div key={index} className="w-20 h-20 rounded-[1.5rem] bg-slate-100" />
                        ))}
                      </div>
                      <div className="rounded-[2rem] border-2 border-dashed border-amber-200 bg-amber-50/70 p-6 text-center">
                        <p className="text-xl font-bold text-amber-700">
                          뜻은 먼저 보여주고, 글자별 풀이를 이어서 불러오고 있어요.
                        </p>
                      </div>
                    </div>
                  )}

                  {!isLoadingDetails && detailError && !hasHanja && !result.syllables && (
                    <div className="rounded-[2rem] border border-amber-200 bg-amber-50 p-6 text-amber-800 font-bold text-lg">
                      지금은 글자별 풀이를 준비하지 못했어요. 다시 검색하면 캐시된 뜻은 더 빨리 볼 수 있어요.
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
                                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30 scale-110 ring-4 ring-blue-200'
                                    : 'bg-blue-50 text-blue-600 hover:bg-blue-100 hover:scale-105 cursor-pointer shadow-sm border-2 border-blue-200'
                              }`}
                            >
                              {syllable.char}
                            </button>
                            {index < result.syllables.length - 1 && (
                              <span className="text-slate-300 font-black text-4xl">+</span>
                            )}
                          </React.Fragment>
                        ))}
                      </div>

                      {!selectedSyllable && (
                        <div className="rounded-[2rem] border-2 border-dashed border-slate-200 bg-slate-50 p-6 text-center text-slate-600 text-lg font-bold">
                          파란 글자를 눌러서 어떤 한자가 들어 있는지 알아보세요.
                        </div>
                      )}

                      <AnimatePresence mode="wait">
                        {selectedSyllable && (
                          <motion.div
                            key={selectedSyllable.char}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="mt-6 rounded-[2rem] border-2 border-amber-200 bg-amber-50/80 p-6"
                          >
                            <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
                              <div>
                                <p className="text-sm font-bold uppercase tracking-[0.2em] text-amber-600 mb-2">
                                  Selected
                                </p>
                                <div className="flex items-end gap-3">
                                  <span className="text-5xl font-black text-slate-900">
                                    {selectedSyllable.char}
                                  </span>
                                  {selectedSyllable.hanjaChar && (
                                    <span className="text-3xl font-black text-amber-700">
                                      {selectedSyllable.hanjaChar}
                                    </span>
                                  )}
                                </div>
                              </div>

                              {selectedSyllable.relatedWords?.length ? (
                                <div className="flex flex-wrap justify-end gap-2">
                                  {selectedSyllable.relatedWords.map((word, index) => (
                                    <span
                                      key={`${word}-${index}`}
                                      className="px-4 py-2 rounded-full bg-white border border-amber-200 text-slate-700 font-bold"
                                    >
                                      {word}
                                    </span>
                                  ))}
                                </div>
                              ) : null}
                            </div>

                            <p className="text-2xl font-bold text-amber-800 mb-6">
                              이 글자에 공통으로 들어 있는 뜻이 무엇일까요?
                            </p>

                            {!showHint ? (
                              <button
                                onClick={() => setShowHint(true)}
                                className="px-8 py-4 bg-amber-400 hover:bg-amber-500 text-white font-black text-2xl rounded-full transition-all shadow-md hover:shadow-lg flex items-center gap-2 transform hover:-translate-y-1"
                              >
                                정답 확인하기 <ChevronRight className="w-6 h-6" />
                              </button>
                            ) : (
                              <motion.div
                                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                className="px-8 py-6 bg-blue-500 text-white font-black text-3xl rounded-[2rem] shadow-lg border-4 border-blue-400 text-center"
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
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
